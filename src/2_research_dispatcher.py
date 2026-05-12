import os
import json
import time
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from graph_agent import app

print("\n" + "="*60)
print("🧠 PHASE 2: THE RESEARCH DISPATCHER (EXPAND & RAG) 🧠")
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

from token_tracker import TokenTrackerCallback

tracker = TokenTrackerCallback(script_name="2_research_dispatcher")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, max_retries=5, callbacks=[tracker])

generated_notes = {}
if os.path.exists("raw_research_notes.json"):
    try:
        with open("raw_research_notes.json", "r", encoding="utf-8") as f:
            generated_notes = json.load(f)
        print("Found previous save state! Resuming research...")
    except Exception as e:
        print(f"Could not load previous save state: {e}. Starting fresh.")

# 2. Iterate through the Blueprint
for section_name, micro_topics in blueprint.items():
    print(f"\nProcessing Section: {section_name}")
    if section_name not in generated_notes:
        generated_notes[section_name] = []
    
    for topic in micro_topics:
        # Check if this topic has already been processed
        already_processed = any(item.get("original_target") == topic for item in generated_notes[section_name])
        if already_processed:
            print(f"\n  Skipping (Already done): {topic[:60]}...")
            continue

        print(f"\n   Target: {topic[:60]}...")
        
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
            expanded_requirements = llm.invoke([expander_prompt, expansion_msg]).content
            print(f"      [Expanded] -> Ready. Dispatching to Graph Agent...")
        except Exception as e:
             print(f"      ❌ Error during expansion: {e}")
             continue 
        
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
            # We use a high recursion limit because the agent may need to use multiple search tools
            final_state = app.invoke(initial_state, config={"recursion_limit": 25})
            final_content = final_state["messages"][-1].content
            print("      [Writer] -> Graph Agent Draft complete.")
        except Exception as e:
            print(f"      ❌ Error during Graph Agent execution: {e}")
            continue

        generated_notes[section_name].append({
            "original_target": topic,
            "expanded_requirements": expanded_requirements,
            "content": final_content
        })
        
        # Save state after EVERY successful generation
        with open("raw_research_notes.json", "w", encoding="utf-8") as f:
            json.dump(generated_notes, f, indent=4)

print("\n✅ All research complete! Safely saved to 'raw_research_notes.json'")