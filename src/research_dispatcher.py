import os
import json
import time
import asyncio
import aiofiles
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from src.graph_agent import app
from src.token_tracker import TokenTrackerCallback

async def run(event_name: str, blueprint: dict, cache_info: dict) -> dict:
    print("\n" + "="*60)
    print("🧠 PHASE 2: THE RESEARCH DISPATCHER (PARALLEL RAG) 🧠")
    print("="*60)

    load_dotenv()
    
    if not blueprint:
        print("❌ Error: No blueprint provided. Please run Phase 1 first.")
        return {}

    tracker = TokenTrackerCallback(script_name="2_research_dispatcher")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, max_retries=5, callbacks=[tracker])

    CONCURRENCY_LIMIT = 5
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    DATA_DIR = "pipeline_data"
    os.makedirs(DATA_DIR, exist_ok=True)
    NOTES_FILE = os.path.join(DATA_DIR, "raw_research_notes.json")

    async def process_topic(section_name, topic, generated_notes_lock, generated_notes):
        async with semaphore:
            print(f"\n   [Start] Target: {topic[:60]}...")
            
            expander_prompt = SystemMessage(content="""You are a Science Olympiad Technical Analyst.
            Take the provided cheat sheet target and expand it into a highly detailed list of exact facts needed.
            DO NOT answer the questions or provide the facts. Just list the specific variables, formulas, definitions, and traps that a researcher needs to go find.
            Since we need extreme depth, generate 5-6 highly specific bullet points.
            Example Output:
            - Formula for X including variables Y and Z.
            - The specific unit conversion mistake often made between metric and imperial for X.
            - The definition of condition A as it relates to X.
            - The derivation or secondary application of X in extreme edge cases.""")
            
            expansion_msg = HumanMessage(content=f"Expand this target: '{topic}'")
            
            try:
                expanded_response = await llm.ainvoke([expander_prompt, expansion_msg])
                expanded_requirements = expanded_response.content
                print(f"      [Expanded] -> {topic[:20]}... Ready. Dispatching to Graph Agent...")
            except Exception as e:
                 print(f"      ❌ Error during expansion for '{topic[:20]}': {e}")
                 return 
            
            system_prompt = SystemMessage(content=f"""You are an expert Science Olympiad AI Assistant building a dense cheat sheet. 
            The user is studying for the event: {event_name}.

            STRICT WORKFLOW PROTOCOL:
            1. SCOPE CHECK: If out of scope, call 'reject_out_of_scope'.
            2. CHECK RULES: Always use 'search_scioly_rules' first. You MUST pass "{event_name}" into the event_metadata parameter.
            3. REQUEST CLEARANCE: If you need web research, use 'request_search_clearance'.
            4. EXTERNAL RESEARCH PRIORITY: 
               - ALWAYS prioritize using 'search_arxiv' for advanced theory.
               - ONLY use the other web sniper tools if ArXiv returns no useful results.
               - BE EFFICIENT: Do not spam search tools. Execute ONE highly targeted search at a time.
            5. GATEKEEPER CHECK: Use 'submit_final_answer' to self-grade.
            6. FINAL OUTPUT FORMAT:
               - Write EXACTLY 130-140 words.
               - Format using dense bullet points.
               - Use bold text for key terms.
               - Include formulas, key stats, edge cases, and precise conditions.
               - ALWAYS highlight test traps or common mistakes.
               - Start immediately with facts. NO conversational filler.
            """)

            research_task = HumanMessage(content=f"""
            ORIGINAL TARGET: {topic}
            
            EXPANDED REQUIREMENTS TO COVER: 
            {expanded_requirements}
            """)

            initial_state = {"messages": [system_prompt, research_task]}

            try:
                final_state = await app.ainvoke(initial_state, config={"recursion_limit": 25})
                final_content = final_state["messages"][-1].content
                print(f"      [Writer] -> Graph Agent Draft complete for: {topic[:20]}...")
            except Exception as e:
                print(f"      ❌ Error during Graph Agent execution for '{topic[:20]}': {e}")
                return

            async with generated_notes_lock:
                if section_name not in generated_notes:
                    generated_notes[section_name] = []
                    
                generated_notes[section_name].append({
                    "original_target": topic,
                    "expanded_requirements": expanded_requirements,
                    "content": final_content
                })
                
                async with aiofiles.open(NOTES_FILE, mode="w", encoding="utf-8") as f:
                    await f.write(json.dumps(generated_notes, indent=4))

    generated_notes = {}
    if os.path.exists(NOTES_FILE):
        try:
            with open(NOTES_FILE, "r", encoding="utf-8") as f:
                generated_notes = json.load(f)
            print("Found previous save state! Resuming research...")
        except Exception as e:
            print(f"Could not load previous save state: {e}. Starting fresh.")

    tasks = []
    generated_notes_lock = asyncio.Lock()
    
    for section_name, micro_topics in blueprint.items():
        print(f"\nQueueing Section: {section_name}")
        if section_name not in generated_notes:
            generated_notes[section_name] = []
        
        for topic in micro_topics:
            already_processed = any(item.get("original_target") == topic for item in generated_notes.get(section_name, []))
            if already_processed:
                print(f"  Skipping (Already done): {topic[:60]}...")
                continue
            
            tasks.append(process_topic(section_name, topic, generated_notes_lock, generated_notes))

    if tasks:
        print(f"\n🚀 Firing {len(tasks)} tasks in parallel (Max Concurrency: {CONCURRENCY_LIMIT})...")
        await asyncio.gather(*tasks)
    else:
        print("\n✅ All topics were already completed!")

    print("\n✅ All research complete! Safely saved to 'pipeline_data/raw_research_notes.json'")
    return generated_notes

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run("Science Olympiad", {}, {}))
