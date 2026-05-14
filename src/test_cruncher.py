import os
import sys
import json
from collections import Counter
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from src.factory import factory, console
from src.models import ExtractedConcepts
from rich.table import Table

def run() -> dict:
    config = factory.get_config()
    DATA_DIR = config['paths']['data_dir']
    TESTS_FOLDER = config['paths']['raw_tests']
    PROGRESS_FILE = os.path.join(DATA_DIR, "cruncher_save_state.json")
    os.makedirs(DATA_DIR, exist_ok=True)

    console.print("\n[phase]PHASE 0: PAST TEST FREQUENCY CRUNCHER[/phase]")

    llm = factory.get_llm(purpose="researcher")
    structured_llm = llm.with_structured_output(ExtractedConcepts)

    cruncher_prompt = SystemMessage(content="""You are a data-mining AI. 
    Read the provided Science Olympiad test questions. Extract highly specific event topics being tested. 
    Standardize the names of the concepts.""")

    if not os.path.exists(TESTS_FOLDER):
        os.makedirs(TESTS_FOLDER)
        console.print(f"[warning]Created '{TESTS_FOLDER}' folder. Drop your .pdf or .docx test files in there and run again.[/warning]")
        return {}

    test_files = [f for f in os.listdir(TESTS_FOLDER) if f.endswith('.pdf') or f.endswith('.docx')]

    if not test_files:
        console.print(f"[warning]No .pdf or .docx files found in '{TESTS_FOLDER}'. Skipping test crunching.[/warning]")
        return {}

    master_concept_list = []
    master_trap_list = []
    processed_files = []

    if os.path.exists(PROGRESS_FILE):
        console.print(f"[info]Found save file! Loading previous progress from '{PROGRESS_FILE}'...[/info]")
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            save_data = json.load(f)
            master_concept_list = save_data.get("concepts", [])
            master_trap_list = save_data.get("traps", [])
            processed_files = save_data.get("processed_files", [])
        
        files_left = len(test_files) - len(processed_files)
        console.print(f"[success]✅ Resuming... {len(processed_files)} files already crunched. {files_left} left to go![/success]\n")
    else:
        console.print(f"[info]📚 Found {len(test_files)} tests. Starting fresh crunch...[/info]\n")

    invoke_with_retry = factory.get_retry_decorator(
        before_sleep_func=lambda retry_state: console.print(f"[warning]⚠️ Rate limit hit. Retrying in {retry_state.next_action.sleep}s...[/warning]")
    )(structured_llm.invoke)

    for filename in test_files:
        if filename in processed_files:
            continue 

        console.print(f"   [cyan]Scanning file:[/cyan] {filename}...")
        
        try:
            filepath = os.path.join(TESTS_FOLDER, filename)
            if filename.endswith('.pdf'):
                loader = PyPDFLoader(filepath)
            elif filename.endswith('.docx'):
                loader = Docx2txtLoader(filepath)
            else:
                continue
                
            pages = loader.load()
            test_content = "\n".join([page.page_content for page in pages])
            test_content = test_content[:100000] 
            
            messages = [cruncher_prompt, HumanMessage(content=f"Extract concepts from this test:\n\n{test_content}")]
            result = invoke_with_retry(messages)
            
            master_concept_list.extend(result.core_concepts)
            master_trap_list.extend(result.test_traps)
            processed_files.append(filename)
            
            with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "concepts": master_concept_list,
                    "traps": master_trap_list,
                    "processed_files": processed_files
                }, f, indent=4)
                
        except Exception as e:
            console.print(f"[error]   ❌ Failed to parse {filename}: {e}[/error]")

    console.print("\n[info]All files processed! Tallying final frequencies...[/info]")
    concept_counts = Counter(master_concept_list)
    trap_counts = Counter(master_trap_list)

    # Use Rich Table for output
    table = Table(title="Test Frequency Leaderboard", show_header=True, header_style="bold cyan")
    table.add_column("Rank", style="dim", width=6)
    table.add_column("Top Core Concepts", style="bold white")
    table.add_column("Top Test Traps", style="yellow")
    
    top_concepts = concept_counts.most_common(20)
    top_traps = trap_counts.most_common(20)
    
    for i in range(20):
        concept = f"{top_concepts[i][0]} ({top_concepts[i][1]}x)" if i < len(top_concepts) else ""
        trap = f"{top_traps[i][0]} ({top_traps[i][1]}x)" if i < len(top_traps) else ""
        table.add_row(str(i+1), concept, trap)
    
    console.print(table)

    frequency_report = {
        "Top_50_Tested_Concepts": [f"{concept} (Tested {count} times)" for concept, count in concept_counts.most_common(50)],
        "Top_20_Test_Traps": [f"{trap} (Seen {count} times)" for trap, count in trap_counts.most_common(20)]
    }

    with open(os.path.join(DATA_DIR, "test_frequency_map.json"), "w", encoding="utf-8") as f:
        json.dump(frequency_report, f, indent=4)

    console.print("[success]✅ SUCCESS! Master frequency leaderboard computed.[/success]")
    return frequency_report

if __name__ == "__main__":
    run()
