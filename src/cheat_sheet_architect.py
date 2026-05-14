import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from src.token_tracker import TokenTrackerCallback

# ==========================================
# 0. SETUP & LOAD SECRETS
# ==========================================
load_dotenv()

class Section(BaseModel):
    section_name: str = Field(description="A dynamically generated name for this section based on the event's rules and core topics.")
    micro_topics: list[str] = Field(description="Exactly 10 hyper-specific search targets. Targets must be complex enough to generate a 130-word response (e.g., asking for facts, formulas, edge cases, and test traps).")

class CheatSheetBlueprint(BaseModel):
    event_analysis: str = Field(description="A 2-sentence analysis of the absolute highest-yield concepts based on the provided rules and frequency leaderboard.")
    sections: list[Section] = Field(description="Exactly 5 major sections to perfectly map to a 50-target, 6500-word physical cheat sheet layout.")

def run(event_name: str, frequency_data: dict) -> tuple[str, dict]:
    print("\n" + "="*60)
    print("ADAPTIVE CHEAT SHEET ARCHITECT 3.0 (DUAL-INPUT)")
    print("="*60)

    # --- THE RULEBOOK INJECTOR ---
    print(f"\nFetching official rules for '{event_name}' from the database...")
    official_rules_text = ""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        vectorstore = Chroma(persist_directory="./scioly_db", embedding_function=embeddings)
        
        try:
            rule_docs = vectorstore.similarity_search(event_name, k=5, filter={"Event": event_name.title()})
        except:
            rule_docs = []
            
        if not rule_docs:
            rule_docs = vectorstore.similarity_search(f"{event_name} rules", k=5)
            
        official_rules_text = "\n\n".join([doc.page_content for doc in rule_docs])
        if not official_rules_text.strip():
             official_rules_text = "No official rules found in the database. Rely on general knowledge for this event."
             print("⚠️ No rules found in DB. Relying on baseline knowledge.")
        else:
             print("✅ Successfully retrieved official rules from database!")
    except Exception as e:
        print(f"⚠️ Warning: Could not load local database for rules. Details: {e}")
        official_rules_text = "Database not accessible. Rely on general knowledge."

    # --- THE LEADERBOARD INJECTOR ---
    if frequency_data:
        test_context = json.dumps(frequency_data, indent=2)
        print("✅ Received Frequency Leaderboard! Architect is using hyper-optimized test data.")
    else:
        print("⚠️ No test frequency data provided. Proceeding with baseline AI knowledge.")
        test_context = "No specific test frequency data provided. Rely on standard national-level Science Olympiad meta."

    print(f"\nWaking up the Architect to isolate the top 50 max-density targets for: {event_name}...")

    tracker = TokenTrackerCallback(script_name="1_cheat_sheet_architect")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1, max_retries=3, callbacks=[tracker])
    structured_llm = llm.with_structured_output(CheatSheetBlueprint)

    prompt_text = f"""You are an elite Science Olympiad National Head Coach for the event: {event_name}.
    You are designing the blueprint for an ultra-dense, competition-dominating cheat sheet.

    YOUR ONLY JOB IS TO BE THE PLANNER. 
    Do not write the actual notes or formulas. You are generating the specific targets that a secondary AI research agent will look up later.

    I am providing you with two critical pieces of data:

    1. THE OFFICIAL RULES:
    {official_rules_text}

    2. PAST TEST FREQUENCY LEADERBOARD:
    {test_context}

    THE MAX DENSITY MATH:
    We have space for exactly 50 targets (5 sections of 10 targets each). The secondary AI will write exactly 130 words per target. 
    Your targets must be incredibly "meaty" and detailed. Do not just ask for a basic fact; ask for the fact, the specific applications, the exact equations, AND the test trap in the same target string.

    CRITICAL CONSTRAINTS & MANDATES:
    1. DYNAMIC SECTIONS: You MUST generate exactly 5 sections. You must dynamically determine the 5 most appropriate section titles based on the official rules. For example, if it's a lab event, include a 'Lab Techniques' section. If it's a biology event, include an 'Anatomy' section.
    2. EXHAUSTIVE RULE MANDATE (NO TRASH INFO): You MUST hunt down and ensure that every single specific item, formula, chemical, plant, animal, and environmental scenario (like toxic spills) explicitly listed in the Official Rules gets its own dedicated target. 
    3. CROSS-REFERENCE MANDATE: You will receive a Past Test Frequency Leaderboard. You MUST filter it. If a leaderboard topic is a broad, generic science concept (like 'Electron Configurations' or 'Basic Stoichiometry') that is NOT explicitly required by the Official Rules, you MUST discard it. Only use leaderboard topics that directly align with the specific event syllabus.
    4. ELIMINATE FLUFF: DO NOT generate targets for general, broad, or high-level scientific concepts (e.g., 'General Atomic Structure', 'Basic VSEPR Theory') unless they are explicitly requested by the rules. If an item is not in the rules or the filtered leaderboard, DO NOT include it. Stick strictly to the event's specific syllabus!
    5. HYPER-SPECIFICITY: The research agent needs exact instructions. 
       - BAD TARGET: "Friction concepts"
       - GOOD TARGET: "The exact formulas for static and kinetic friction, the derivation of mu from an inclined plane angle, and the trap of confusing mass with normal force when calculating acceleration."
    """

    architect_prompt = SystemMessage(content=prompt_text)

    messages = [
        architect_prompt, 
        HumanMessage(content=f"Generate the highly constrained 50-topic max-density blueprint for {event_name} based on the rules and leaderboard.")
    ]

    try:
        print("🧠 Architect is analyzing the rules, leaderboard, and calculating physical space...")
        blueprint = structured_llm.invoke(messages)
        
        print(f"\nMeta Analysis: {blueprint.event_analysis}\n")
        
        final_dict = {}
        total_targets = 0
        for sec in blueprint.sections:
            final_dict[sec.section_name.replace(" ", "_")] = sec.micro_topics
            total_targets += len(sec.micro_topics)
        
        # We can still dump to json for debugging/checkpointing
        DATA_DIR = "pipeline_data"
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(os.path.join(DATA_DIR, "cheat_sheet_blueprint.json"), "w", encoding="utf-8") as f:
            json.dump(final_dict, f, indent=4)
            
        print(f"✅ SUCCESS! Master Blueprint generated.")
        print(f"Total Search Targets Planned: {total_targets} (Perfectly calibrated for 6500+ words of ultra-dense text)")
        return event_name, final_dict
            
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        return event_name, {}

if __name__ == "__main__":
    event = input("\nWhat Science Olympiad event are you building a cheat sheet for? ")
    run(event, {})
