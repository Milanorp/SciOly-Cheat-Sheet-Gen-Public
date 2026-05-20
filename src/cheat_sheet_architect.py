import os
import json
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from src.token_tracker import TokenTrackerCallback
from src.factory import factory
from src.models import CheatSheetBlueprint

def run(event_name: str, frequency_data: dict) -> tuple[str, dict]:
    config = factory.get_config()
    DATA_DIR = config['paths']['data_dir']
    os.makedirs(DATA_DIR, exist_ok=True)

    print("\n" + "="*60)
    print("ADAPTIVE CHEAT SHEET ARCHITECT 3.0 (DUAL-INPUT)")
    print("="*60)

    # --- THE RULEBOOK INJECTOR ---
    print(f"\nFetching official rules for '{event_name}' from the database...")
    official_rules_text = ""
    try:
        db_path = os.path.abspath(config['database']['db_path'])
        if not os.path.exists(db_path):
             print(f"⚠️ Warning: Database folder not found at {db_path}. Please run 'python src/build_db.py' first.")
        
        embeddings = factory.get_embeddings()
        vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
        
        # Try 1: Specific Filter
        try:
            rule_docs = vectorstore.similarity_search(event_name, k=5, filter={"Event": event_name.title()})
        except:
            rule_docs = []
            
        # Try 2: Loose search (no filter) if Try 1 failed
        if not rule_docs:
            print("   > Specific filter failed. Trying a broad search...")
            rule_docs = vectorstore.similarity_search(f"{event_name} rules", k=8)
            
        official_rules_text = "\n\n".join([doc.page_content for doc in rule_docs])
        if not official_rules_text.strip():
             official_rules_text = "No official rules found in the database. Rely on general knowledge for this event."
             print("⚠️ No rules found in DB. Relying on baseline knowledge.")
        else:
             print(f"✅ Successfully retrieved {len(rule_docs)} rule segments from database!")
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

    print(f"\nWaking up the Architect to isolate the top {config['research']['total_targets']} max-density targets for: {event_name}...")

    tracker = TokenTrackerCallback(script_name="1_cheat_sheet_architect")
    llm = factory.get_llm(purpose="architect")
    # Bind tracker manually since factory handles global setup
    llm.callbacks = [tracker]
    
    structured_llm = llm.with_structured_output(CheatSheetBlueprint)

    prompt_text = f"""You are an elite Science Olympiad National Head Coach for the event: {event_name}.
    You are designing the blueprint for an ultra-dense, competition-dominating cheat sheet.

    YOUR ONLY JOB IS TO BE THE PLANNER. 
    Do not write the actual notes or formulas. You are generating the specific targets that a secondary AI research agent will look up later.

    I am providing you with two critical pieces of data:

    1. THE OFFICIAL RULES:
    {{official_rules_text}}

    2. PAST TEST FREQUENCY LEADERBOARD & EXAMPLES:
    {{test_context}}

    THE MAX DENSITY MATH:
    We have space for exactly {config['research']['total_targets']} targets ({config['research']['sections_count']} sections of {config['research']['targets_per_section']} targets each). 
    The secondary AI will write exactly {config['research']['target_word_count']} words per target. 
    Your targets must be incredibly "meaty" and detailed. Do not just ask for a basic fact; ask for the fact, the specific applications, the exact equations, AND the test trap in the same target string.

    CRITICAL CONSTRAINTS & MANDATES:
    1. DYNAMIC SECTIONS: You MUST generate exactly {config['research']['sections_count']} sections. You must dynamically determine the {config['research']['sections_count']} most appropriate section titles based on the official rules.
    2. EXHAUSTIVE RULE MANDATE: You MUST hunt down and ensure that every single specific item, formula, chemical, plant, animal, and environmental scenario explicitly listed in the Official Rules gets its own dedicated target. 
    3. THE EXAMPLE MANDATE: You MUST look at the 'High Impact Examples' provided in the leaderboard. You MUST generate at least 5 targets specifically designed to help solve those exact difficult problem types. 
    4. CROSS-REFERENCE MANDATE: You will receive a Past Test Frequency Leaderboard. You MUST filter it. Only use leaderboard topics that directly align with the specific event syllabus.
    5. ELIMINATE FLUFF: DO NOT generate targets for general, broad, or high-level scientific concepts unless they are explicitly requested by the rules. 
    6. HYPER-SPECIFICITY: The research agent needs exact instructions. 
    """

    architect_prompt = SystemMessage(content=prompt_text)

    messages = [
        architect_prompt, 
        HumanMessage(content=f"Generate the highly constrained blueprint for {event_name} based on the rules and leaderboard.")
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
        with open(os.path.join(DATA_DIR, "cheat_sheet_blueprint.json"), "w", encoding="utf-8") as f:
            json.dump(final_dict, f, indent=4)
            
        print(f"✅ SUCCESS! Master Blueprint generated.")
        print(f"Total Search Targets Planned: {total_targets} (Perfectly calibrated for {total_targets * config['research']['target_word_count']} words of ultra-dense text)")
        return event_name, final_dict
            
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        return event_name, {}

if __name__ == "__main__":
    event = input("\nWhat Science Olympiad event are you building a cheat sheet for? ")
    run(event, {})
