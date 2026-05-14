import os
import json
import time
import asyncio
import aiofiles
from langchain_core.messages import HumanMessage, SystemMessage
from src.graph_agent import app
from src.token_tracker import TokenTrackerCallback
from src.factory import factory, console
from src.models import ResearchNote
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, MofNCompleteColumn

async def run(event_name: str, blueprint: dict, cache_info: dict, target_topics: list = None) -> dict:
    config = factory.get_config()
    DATA_DIR = config['paths']['data_dir']
    NOTES_FILE = os.path.join(DATA_DIR, "raw_research_notes.json")
    os.makedirs(DATA_DIR, exist_ok=True)

    console.print("\n[phase]🧠 PHASE 2: THE RESEARCH DISPATCHER (PARALLEL RAG) 🧠[/phase]")

    tracker = TokenTrackerCallback(script_name="2_research_dispatcher")
    llm = factory.get_llm(purpose="researcher")
    llm.callbacks = [tracker]

    CONCURRENCY_LIMIT = config['research']['concurrency_limit']
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async def process_topic(section_name, topic, generated_notes_lock, generated_notes, progress, task_id):
        async with semaphore:
            # console.print(f"   [info][Start] Target:[/info] {topic[:50]}...")
            
            expander_prompt = SystemMessage(content="""You are a Science Olympiad Technical Analyst.
            List specific variables, formulas, definitions, and traps for the provided topic. 5-6 bullet points.""")
            
            try:
                expanded_response = await llm.ainvoke([expander_prompt, HumanMessage(content=f"Expand: '{topic}'")])
                expanded_requirements = expanded_response.content
                
                system_prompt = SystemMessage(content=f"""You are an expert Science Olympiad AI Assistant building a dense cheat sheet for {event_name}.
                STRICT WORKFLOW: 1. Rules search, 2. ArXiv, 3. Sniper tools.
                Output: EXACTLY {config['research']['target_word_count']} words, bullet points, LaTeX math.""")

                research_task = HumanMessage(content=f"TARGET: {topic}\nREQUIREMENTS: {expanded_requirements}")
                initial_state = {"messages": [system_prompt, research_task]}

                final_state = await app.ainvoke(initial_state, config={"recursion_limit": 25})
                final_content = final_state["messages"][-1].content

                async with generated_notes_lock:
                    if section_name not in generated_notes:
                        generated_notes[section_name] = []
                    
                    existing_entry_idx = next((i for i, item in enumerate(generated_notes[section_name]) if item.get("original_target") == topic), None)
                    new_entry = {"original_target": topic, "expanded_requirements": expanded_requirements, "content": final_content}

                    if existing_entry_idx is not None:
                        generated_notes[section_name][existing_entry_idx] = new_entry
                    else:
                        generated_notes[section_name].append(new_entry)
                    
                    async with aiofiles.open(NOTES_FILE, mode="w", encoding="utf-8") as f:
                        await f.write(json.dumps(generated_notes, indent=4))
                
                progress.update(task_id, advance=1, description=f"[success]Completed:[/success] {topic[:30]}...")
            except Exception as e:
                console.print(f"[error]      ❌ Error for '{topic[:20]}': {e}[/error]")
                progress.update(task_id, description=f"[error]Failed:[/error] {topic[:30]}...")

    generated_notes = {}
    if os.path.exists(NOTES_FILE):
        try:
            with open(NOTES_FILE, "r", encoding="utf-8") as f:
                generated_notes = json.load(f)
            console.print("[info]Found previous save state! Resuming research...[/info]")
        except:
            pass

    # Collect all topics to process
    all_tasks_data = []
    for section_name, micro_topics in blueprint.items():
        for topic in micro_topics:
            if target_topics and topic not in target_topics:
                continue
            if not target_topics:
                already_processed = any(item.get("original_target") == topic for item in generated_notes.get(section_name, []))
                if already_processed:
                    continue
            all_tasks_data.append((section_name, topic))

    if all_tasks_data:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            console=console
        ) as progress:
            master_task = progress.add_task("[cyan]Researching Topics...", total=len(all_tasks_data))
            
            generated_notes_lock = asyncio.Lock()
            tasks = [process_topic(sec, top, generated_notes_lock, generated_notes, progress, master_task) for sec, top in all_tasks_data]
            await asyncio.gather(*tasks)
    else:
        console.print("[success]✅ No research tasks needed in this run![/success]")

    console.print(f"\n[success]✅ Phase 2 complete! Safely saved to '{NOTES_FILE}'[/success]")
    return generated_notes

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run("Science Olympiad", {}, {}))
