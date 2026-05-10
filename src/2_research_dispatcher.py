import os
import json
import time
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma 

print("\n" + "="*60)
print("🧠 PHASE 2: THE RESEARCH DISPATCHER (EXPAND & RAG) 🧠")
print("="*60)

load_dotenv()

# 0. Load the Database
print("Loading Local Vector Database...")
try:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = Chroma(persist_directory="./scioly_db", embedding_function=embeddings)
except Exception as e:
    print(f"❌ Error loading database. Did you run 'build_db.py' first? Details: {e}")
    exit(1)

# 0.5 Load the Cache
CACHE_NAME = None
if os.path.exists("cache_info.json"):
    try:
        with open("cache_info.json", "r", encoding="utf-8") as f:
            CACHE_NAME = json.load(f).get("cache_name")
        print(f"📦 Loaded Gemini Context Cache: {CACHE_NAME}")
    except Exception as e:
        print(f"⚠️ Could not load context cache: {e}")

# 1. Load the Blueprint
try:
    with open("cheat_sheet_blueprint.json", "r", encoding="utf-8") as f:
        blueprint = json.load(f)
    print("✅ Successfully loaded 'cheat_sheet_blueprint.json'.")
except FileNotFoundError:
    print("❌ Error: 'cheat_sheet_blueprint.json' not found. Please run '1_cheat_sheet_architect.py' first.")
    exit(1)

from token_tracker import TokenTrackerCallback

# Use temperature 0.2 for slight creativity in expansion, but strictness in facts
# max_retries=5 tells LangChain to automatically pause and try again if it hits a 429 error!
# We attach our new TokenTrackerCallback so it "listens" to every API call!
tracker = TokenTrackerCallback(script_name="2_research_dispatcher")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, max_retries=5, callbacks=[tracker])

generated_notes = {}
if os.path.exists("raw_research_notes.json"):
    try:
        with open("raw_research_notes.json", "r", encoding="utf-8") as f:
            generated_notes = json.load(f)
        print("💾 Found previous save state! Resuming research...")
    except Exception as e:
        print(f"⚠️ Could not load previous save state: {e}. Starting fresh.")

# 2. Iterate through the Blueprint
for section_name, micro_topics in blueprint.items():
    print(f"\n📁 Processing Section: {section_name}")
    if section_name not in generated_notes:
        generated_notes[section_name] = []
    
    for topic in micro_topics:
        # Check if this topic has already been processed in the current section
        already_processed = any(item.get("original_target") == topic for item in generated_notes[section_name])
        if already_processed:
            print(f"\n   ⏭️ Skipping (Already done): {topic[:60]}...")
            continue

        print(f"\n   🔍 Target: {topic[:60]}...")
        
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
            print(f"      [Expanded] -> Ready. Querying DB...")
        except Exception as e:
             print(f"      ❌ Error during expansion: {e}")
             continue # Skip to the next target if this one fails
        
        # ==========================================
        # STEP 2: RAG SEARCH USING EXPANDED TEXT
        # ==========================================
        # Search the DB using the EXPANDED text for better semantic matching. Increase k to 6 for more depth.
        try:
            db_results = vectorstore.similarity_search(expanded_requirements, k=6)
            # Combine the DB results into a single context string
            rag_context = "\n\n---\n\n".join([doc.page_content for doc in db_results])
            if not rag_context.strip():
                 rag_context = "No specific rules found in the local database for this exact topic."
        except Exception as e:
             print(f"      ⚠️ Database search failed: {e}")
             rag_context = "Database search failed."
        
        # ==========================================
        # STEP 3: THE WRITER PHASE
        # ==========================================
        final_prompt_text = f"""You are an expert Science Olympiad note-taker building a dense cheat sheet. 
        Write EXACTLY 130-140 words to fulfill the requested target with extreme technical depth.
        
        STRICT RULES:
        1. Use the provided Rulebook Context AND the Cached Rulebook (if available) to ensure your facts are accurate.
        2. Format using dense bullet points. Use bold text for key terms.
        3. Include formulas, key stats, edge cases, and precise conditions.
        4. ALWAYS highlight test traps or common mistakes.
        5. Start immediately with facts. NO conversational filler.
        
        ORIGINAL TARGET: {topic}
        
        EXPANDED REQUIREMENTS TO COVER: 
        {expanded_requirements}
        
        RAG RULEBOOK CONTEXT:
        {rag_context}
        """

        try:
            if CACHE_NAME:
                from google import genai
                from google.genai import types
                genai_client = genai.Client()
                
                response = genai_client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=final_prompt_text,
                    config=types.GenerateContentConfig(
                        cached_content=CACHE_NAME,
                        temperature=0.2
                    )
                )
                final_content = response.text
                
                # Manually log tokens to the tracker
                if response.usage_metadata:
                    in_tokens = response.usage_metadata.prompt_token_count
                    out_tokens = response.usage_metadata.candidates_token_count
                    tracker._log_to_file(in_tokens, out_tokens)
                    print(f"      [Writer] -> Draft complete. (Cached Cost: {in_tokens} tokens)")
                else:
                    print("      [Writer] -> Draft complete. (Cached)")
            else:
                writer_prompt = SystemMessage(content="You are an expert Science Olympiad note-taker. Follow the prompt instructions perfectly.")
                final_req = HumanMessage(content=final_prompt_text)
                final_note = llm.invoke([writer_prompt, final_req])
                final_content = final_note.content
                print("      [Writer] -> Draft complete. (Standard LangChain)")
                
            generated_notes[section_name].append({
                "original_target": topic,
                "expanded_requirements": expanded_requirements,
                "content": final_content
            })
            
            # Save state after EVERY successful generation so we don't lose progress if API fails later
            with open("raw_research_notes.json", "w", encoding="utf-8") as f:
                json.dump(generated_notes, f, indent=4)
                
            # API Rate limit protection removed - user is on Paid Tier!
        except Exception as e:
            print(f"      ❌ Error writing final note: {e}")

print("\n✅ All research complete! Safely saved to 'raw_research_notes.json'")