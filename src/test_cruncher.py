import os
import sys
import json
from collections import Counter
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
import tenacity

# ==========================================
# 0. SETUP & LOAD SECRETS
# ==========================================
load_dotenv()

class ExtractedConcepts(BaseModel):
    core_concepts: list[str] = Field(description="A list of the specific concepts, formulas, diseases, or structures tested in this document. Keep them short (e.g., 'Glomerular Filtration Rate', 'Class 2 Levers').")
    test_traps: list[str] = Field(description="A list of tricky wordings, unit conversions, or common mistakes found in these questions.")

def is_rate_limit(exception):
    err_str = str(exception).lower()
    return "429" in err_str or "quota" in err_str or "exhausted" in err_str

def run() -> dict:
    print("\n" + "="*60)
    print("PHASE 0: PAST TEST FREQUENCY CRUNCHER")
    print("="*60)

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    structured_llm = llm.with_structured_output(ExtractedConcepts)

    cruncher_prompt = SystemMessage(content="""You are a data-mining AI. 
    Read the provided Science Olympiad test questions. Extract highly specific event topics being tested (e.g., specific household toxins, poisonous plants/animals, spill dynamics, exact chemical reactions). 
    DO NOT extract basic, foundational science concepts like 'Atomic Structure', 'VSEPR', or 'Basic Stoichiometry'. Only extract the specific advanced applications and any obvious traps or tricky unit conversions. 
    Standardize the names of the concepts.""")

    TESTS_FOLDER = "raw_tests"
    DATA_DIR = "pipeline_data"
    os.makedirs(DATA_DIR, exist_ok=True)
    PROGRESS_FILE = os.path.join(DATA_DIR, "cruncher_save_state.json")

    if not os.path.exists(TESTS_FOLDER):
        os.makedirs(TESTS_FOLDER)
        print(f"Created '{TESTS_FOLDER}' folder. Drop your .pdf or .docx test files in there and run again.")
        return {}

    test_files = [f for f in os.listdir(TESTS_FOLDER) if f.endswith('.pdf') or f.endswith('.docx')]

    if not test_files:
        print(f"No .pdf or .docx files found in '{TESTS_FOLDER}'. Skipping test crunching.")
        return {}

    master_concept_list = []
    master_trap_list = []
    processed_files = []

    if os.path.exists(PROGRESS_FILE):
        print(f"Found save file! Loading previous progress from '{PROGRESS_FILE}'...")
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            save_data = json.load(f)
            master_concept_list = save_data.get("concepts", [])
            master_trap_list = save_data.get("traps", [])
            processed_files = save_data.get("processed_files", [])
        
        files_left = len(test_files) - len(processed_files)
        print(f"✅ Resuming... {len(processed_files)} files already crunched. {files_left} left to go!\n")
    else:
        print(f"📚 Found {len(test_files)} tests. Starting fresh crunch...\n")

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=2, min=4, max=65),
        stop=tenacity.stop_after_attempt(5),
        retry=tenacity.retry_if_exception(is_rate_limit),
        before_sleep=lambda retry_state: print(f"⚠️ Rate limit hit. Retrying in {retry_state.next_action.sleep} seconds...")
    )
    def invoke_with_retry(messages):
        return structured_llm.invoke(messages)

    for filename in test_files:
        if filename in processed_files:
            continue 

        filepath = os.path.join(TESTS_FOLDER, filename)
        print(f"   Scanning file: {filename}...")
        
        try:
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
            print(f"   ❌ Failed to parse {filename}: {e}")

    print("\nAll files processed! Tallying final frequencies...")
    concept_counts = Counter(master_concept_list)
    trap_counts = Counter(master_trap_list)

    frequency_report = {
        "Top_50_Tested_Concepts": [f"{concept} (Tested {count} times)" for concept, count in concept_counts.most_common(50)],
        "Top_20_Test_Traps": [f"{trap} (Seen {count} times)" for trap, count in trap_counts.most_common(20)]
    }

    with open(os.path.join(DATA_DIR, "test_frequency_map.json"), "w", encoding="utf-8") as f:
        json.dump(frequency_report, f, indent=4)

    print("✅ SUCCESS! Master frequency leaderboard computed.")
    return frequency_report

if __name__ == "__main__":
    run()
