import os
import json
import time
import asyncio
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from graph_agent import app
from token_tracker import TokenTrackerCallback

print("\n" + "="*60)
print("🧠 PHASE 2: THE RESEARCH DISPATCHER (PARALLEL RAG) 🧠")
print("="*60)

load_dotenv()

# Read the event name from Phase 1
try:
    with open("event_name.txt", "r", encoding="utf-8") as f:
        EVENT_NAME = f.read().strip()
except FileNotFoundError:
    EVENT_NAME = "Science Olympiad"

# 1. Load the Blueprint
try:
    with open("cheat_sheet_blueprint.json", "r", encoding="utf-8") as f:
        blueprint = json.load(f)
    print("✅ Successfully loaded 'cheat_sheet_blueprint.json'.")
except FileNotFoundError:
    print("❌ Error: 'cheat_sheet_blueprint.json' not found. Please run '1_cheat_sheet_architect.py' first.")
    exit(1)

tracker = TokenTrackerCallback(script_name="2_research_dispatcher")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, max_retries=5, callbacks=[tracker])

# Concurrency limiting to prevent 429 API rate limits
CONCURRENCY_LIMIT = 5
semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

async def process_topic(section_name, topic, generated_notes_lock):
    async with semaphore:
        print(f"\n   [Start] Target: {topic[:60]}...")
        
        # ==========================================
        # STEP 1: THE EXPANSION PHASE
        # ==========================================
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
        
        # ==========================================
        # STEP 2: GRAPH AGENT RAG
        # ==========================================
        system_prompt = SystemMessage(content=f"""You are an expert Science Olympiad AI Assistant building a dense cheat sheet. 
        The user is studying for the event: {EVENT_NAME}.

        STRICT WORKFLOW PROTOCOL:
        1. SCOPE CHECK: If out of scope, call 'reject_out_of_scope'.
        2. CHECK RULES: Always use 'search_scioly_rules' first. You MUST pass "{EVENT_NAME}" into the event_metadata parameter.
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
            # We use ainvoke for parallel non-blocking execution
            final_state = await app.ainvoke(initial_state, config={"recursion_limit": 25})
            final_content = final_state["messages"][-1].content
            print(f"      [Writer] -> Graph Agent Draft complete for: {topic[:20]}...")
        except Exception as e:
            print(f"      ❌ Error during Graph Agent execution for '{topic[:20]}': {e}")
            return

        async with generated_notes_lock:
            # Safely append and save to disk
            generated_notes = {}
            if os.path.exists("raw_research_notes.json"):
                try:
                    with open("raw_research_notes.json", "r", encoding="utf-8") as f:
                        generated_notes = json.load(f)
                except Exception:
                    pass

            if section_name not in generated_notes:
                generated_notes[section_name] = []
                
            generated_notes[section_name].append({
                "original_target": topic,
                "expanded_requirements": expanded_requirements,
                "content": final_content
            })
            
            with open("raw_research_notes.json", "w", encoding="utf-8") as f:
                json.dump(generated_notes, f, indent=4)


async def main():
    generated_notes = {}
    if os.path.exists("raw_research_notes.json"):
        try:
            with open("raw_research_notes.json", "r", encoding="utf-8") as f:
                generated_notes = json.load(f)
            print("Found previous save state! Resuming research...")
        except Exception as e:
            print(f"Could not load previous save state: {e}. Starting fresh.")

    # 2. Iterate through the Blueprint and gather tasks
    tasks = []
    generated_notes_lock = asyncio.Lock()
    
    for section_name, micro_topics in blueprint.items():
        print(f"\nQueueing Section: {section_name}")
        if section_name not in generated_notes:
            generated_notes[section_name] = []
        
        for topic in micro_topics:
            # Check if this topic has already been processed
            already_processed = any(item.get("original_target") == topic for item in generated_notes.get(section_name, []))
            if already_processed:
                print(f"  Skipping (Already done): {topic[:60]}...")
                continue
            
            tasks.append(process_topic(section_name, topic, generated_notes_lock))

    if tasks:
        print(f"\n🚀 Firing {len(tasks)} tasks in parallel (Max Concurrency: {CONCURRENCY_LIMIT})...")
        await asyncio.gather(*tasks)
    else:
        print("\n✅ All topics were already completed!")

    print("\n✅ All research complete! Safely saved to 'raw_research_notes.json'")

if __name__ == "__main__":
    # Workaround for Windows nested asyncio loops if run in specific environments
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())