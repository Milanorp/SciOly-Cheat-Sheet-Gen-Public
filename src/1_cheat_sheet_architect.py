import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# ==========================================
# 0. SETUP & LOAD SECRETS
# ==========================================
print("\n" + "="*60)
print("🏆 ADAPTIVE CHEAT SHEET ARCHITECT 3.0 (DATA-DRIVEN) 🏆")
print("="*60)

load_dotenv()

EVENT_NAME = input("\n🎯 What Science Olympiad event are you building a cheat sheet for? ")

# --- THE LEADERBOARD INJECTOR ---
test_context = ""
try:
    with open("test_frequency_map.json", "r", encoding="utf-8") as f:
        frequency_data = json.load(f)
        test_context = json.dumps(frequency_data, indent=2)
    print("📈 Found Frequency Leaderboard! Architect is using hyper-optimized test data.")
except FileNotFoundError:
    print("⚠️ No 'test_frequency_map.json' found. Proceeding with baseline AI knowledge.")
    test_context = "No specific test frequency data provided. Rely on standard national-level Science Olympiad meta."

print(f"\n🏗️ Waking up the Architect to isolate the top 50 max-density targets for: {EVENT_NAME}...")

from token_tracker import TokenTrackerCallback

# Low temperature because we want strict, strategic logic
tracker = TokenTrackerCallback(script_name="1_cheat_sheet_architect")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1, max_retries=3, callbacks=[tracker])

# ==========================================
# 1. THE STRICT SPACE-CONSTRAINED STRUCTURE
# ==========================================
class Section(BaseModel):
    section_name: str = Field(
        description="The exact name of the cheat sheet block based on the prompt's structural mandate."
    )
    micro_topics: list[str] = Field(
        description="Exactly 10 hyper-specific search targets. Target must be complex enough to generate a 130-word response (e.g., ask for the fact, disease application, and trap)."
    )

class CheatSheetBlueprint(BaseModel):
    event_analysis: str = Field(
        description="A 2-sentence analysis of the absolute highest-yield concepts based on the provided frequency leaderboard."
    )
    sections: list[Section] = Field(
        description="Exactly 5 major sections to perfectly map to a 50-target, 6500-word physical cheat sheet layout."
    )

structured_llm = llm.with_structured_output(CheatSheetBlueprint)

# ==========================================
# 2. THE BLUEPRINT MASTER PROMPT
# ==========================================
architect_prompt = SystemMessage(content=f"""You are an elite Science Olympiad National Head Coach for the event: {EVENT_NAME}.
You are designing the blueprint for an ultra-dense, competition-dominating cheat sheet.

YOUR ONLY JOB IS TO BE THE PLANNER. 
Do not write the actual notes or formulas. You are generating the specific targets that a secondary AI research agent will look up later.

THE MAX DENSITY MATH:
We have space for exactly 50 targets (5 sections of 10 targets each). The secondary AI will write exactly 130 words per target. 
Your targets must be incredibly "meaty" and detailed. Do not just ask for a basic fact; ask for the fact, the disease application, the exact equations, AND the test trap in the same target string.

CRITICAL CONSTRAINTS:
1. MAXIMIZE YIELD: Space is maximized. Base your targets entirely on the highest-ranking concepts from the provided frequency leaderboard, but explore them with extreme depth.
2. HYPER-SPECIFICITY: The research agent needs exact instructions. 
   - BAD TARGET: "Friction concepts"
   - GOOD TARGET: "The exact formulas for static and kinetic friction, the derivation of mu from an inclined plane angle, and the trap of confusing mass with normal force when calculating acceleration."

THE ELITE STRUCTURAL MANDATE (You MUST generate exactly these 5 sections):
1. ⚡ NORMAL VALUES & CORE MATH (Critical baseline stats, ranges, or core formulas)
2. ⚡ RAPID COMPARISON TABLES (Side-by-side differences of highly tested concepts)
3. ⚡ LABELED DIAGRAM BANK & ID STRATEGIES (Specific structures the secondary AI needs to find images/IDs for)
4. ⚡ DISEASE CONNECTIONS & PATHOPHYSIOLOGY (Applied physiology, machine errors, or specific conditions)
5. ⚡ RAPID LOGIC RULES & TRAPS ("If X -> Then Y" logic and high-yield student mistakes)

PAST TEST FREQUENCY LEADERBOARD:
Here is the exact frequency of concepts and traps extracted from past tests. 
You MUST base your 50 targets heavily on the highest-ranking concepts in this list.
{test_context}
""")

messages = [
    architect_prompt, 
    HumanMessage(content=f"Generate the highly constrained 50-topic max-density blueprint for {EVENT_NAME} based on the leaderboard.")
]

# ==========================================
# 3. GENERATE AND SAVE
# ==========================================
try:
    print("🧠 Architect is analyzing the leaderboard and calculating physical space...")
    blueprint = structured_llm.invoke(messages)
    
    print(f"\n💡 Meta Analysis: {blueprint.event_analysis}\n")
    
    # Restructure into a clean dictionary for the Dispatcher to read later
    final_dict = {}
    total_targets = 0
    for sec in blueprint.sections:
        final_dict[sec.section_name.replace(" ", "_")] = sec.micro_topics
        total_targets += len(sec.micro_topics)
    
    with open("cheat_sheet_blueprint.json", "w", encoding="utf-8") as f:
        json.dump(final_dict, f, indent=4)
        
    print(f"✅ SUCCESS! Master Blueprint saved to 'cheat_sheet_blueprint.json'.")
    print(f"Total Search Targets Planned: {total_targets} (Perfectly calibrated for 6500+ words of ultra-dense text)")
        
except Exception as e:
    print(f"\n❌ FATAL ERROR: {e}")