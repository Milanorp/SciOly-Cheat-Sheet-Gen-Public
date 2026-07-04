import os
import json
import time
import asyncio
import aiofiles
from langchain_core.messages import HumanMessage, SystemMessage
from src.graph_agent import build_app
from src.token_tracker import TokenTrackerCallback
from src.factory import factory, console
from src.models import ResearchNote
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, MofNCompleteColumn

def _cast_to_string(content) -> str:
    if isinstance(content, list):
        final_str = ""
        for item in content:
            if isinstance(item, dict) and "text" in item:
                final_str += item["text"]
            elif isinstance(item, str):
                final_str += item
        return final_str
    return str(content)

async def run(event_name: str, blueprint: dict, cache_info: dict, target_topics: list = None) -> dict:
    config = factory.get_config()
    DATA_DIR = config['paths']['data_dir']
    NOTES_FILE = os.path.join(DATA_DIR, "raw_research_notes.json")
    os.makedirs(DATA_DIR, exist_ok=True)

    console.print("\n[phase]🧠 PHASE 2: THE RESEARCH DISPATCHER (PARALLEL RAG) 🧠[/phase]")

    tracker = TokenTrackerCallback(script_name="2_research_dispatcher")
    llm = factory.get_llm(purpose="researcher", cache_info=cache_info)
    llm.callbacks = [tracker]
    
    app = build_app(cache_info=cache_info)

    CONCURRENCY_LIMIT = config['research']['concurrency_limit']

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

    if not all_tasks_data:
        console.print("[success]✅ No research tasks needed in this run![/success]")
        return generated_notes

    console.print(f"[cyan]Expanding {len(all_tasks_data)} topics in batch...[/cyan]")
    expander_prompt = SystemMessage(content="You are a Science Olympiad Technical Analyst.\nList specific variables, formulas, definitions, and traps for the provided topic. 5-6 bullet points.")
    
    expansion_inputs = [[expander_prompt, HumanMessage(content=f"Expand: '{topic}'")] for sec, topic in all_tasks_data]
    
    # Run the LLM batch
    try:
        expansion_results = await llm.abatch(expansion_inputs, config={"max_concurrency": CONCURRENCY_LIMIT})
    except Exception as e:
        console.print(f"[error]Error during expansion batch: {e}[/error]")
        return generated_notes

    # 2. Run agent on all topics in batch
    console.print(f"[cyan]Running Researcher Agent on {len(all_tasks_data)} topics in batch...[/cyan]")
    agent_inputs = []
    for (sec, topic), res in zip(all_tasks_data, expansion_results):
        expanded_requirements = _cast_to_string(res.content)
        system_prompt = SystemMessage(content=f"""You are an expert Science Olympiad AI Assistant building a dense cheat sheet for {event_name}.
        STRICT WORKFLOW: 
        1. RULES: Use 'search_scioly_rules' to find the official constraints.
        2. PAST TESTS: Use 'search_past_tests' to see exactly how this topic is questioned and what level of detail is needed.
        3. THEORY: Use 'search_arxiv' or Sniper tools for advanced depth.

        Output: EXACTLY {config['research']['target_word_count']} words, bullet points, LaTeX math.
        DENSITY RULES: 
        - Use $...$ for inline math. AVOID block math ($$...$$).
        - Use \\ce{{...}} for ALL chemical formulas (e.g., \\ce{{H2O}}, \\ce{{CO2}}, \\ce{{Na+}}).
        - Ensure units and constants are in math mode.""")
        
        research_task = HumanMessage(content=f"TARGET: {topic}\nREQUIREMENTS: {expanded_requirements}")
        agent_inputs.append({"messages": [system_prompt, research_task]})

    # Run the Agent batch
    try:
        agent_results = await app.abatch(agent_inputs, config={"max_concurrency": CONCURRENCY_LIMIT, "recursion_limit": 12})
    except Exception as e:
        console.print(f"[error]Error during agent batch: {e}[/error]")
        return generated_notes

    # 3. Save Results
    console.print(f"[cyan]Saving {len(all_tasks_data)} completed notes...[/cyan]")
    for (sec, topic), exp_res, agt_res in zip(all_tasks_data, expansion_results, agent_results):
        expanded_requirements = _cast_to_string(exp_res.content)
        
        raw_content = _cast_to_string(agt_res["messages"][-1].content)

        if sec not in generated_notes:
            generated_notes[sec] = []
        
        existing_entry_idx = next((i for i, item in enumerate(generated_notes[sec]) if item.get("original_target") == topic), None)
        
        new_note = ResearchNote(
            original_target=topic,
            expanded_requirements=expanded_requirements,
            content=raw_content,
            is_verified=False
        )

        if existing_entry_idx is not None:
            generated_notes[sec][existing_entry_idx] = new_note.model_dump()
        else:
            generated_notes[sec].append(new_note.model_dump())
            
    with open(NOTES_FILE, "w", encoding="utf-8") as f:
        json.dump(generated_notes, f, indent=4)

    console.print(f"\n[success]✅ Phase 2 complete! Safely saved to '{NOTES_FILE}'[/success]")
    return generated_notes
