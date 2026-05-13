import os
import sys
import json
from collections import Counter
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

# ==========================================
# 0. SETUP & LOAD SECRETS
# ==========================================
print("\n" + "="*60)
print("PHASE 0: PAST TEST FREQUENCY CRUNCHER (CHECKPOINT EDITION)")
print("="*60)

load_dotenv()

# Temperature 0: We want exact concept extraction, no hallucinated topics
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# ==========================================
# 1. THE DATA EXTRACTION STRUCTURE
# ==========================================
class ExtractedConcepts(BaseModel):
    core_concepts: list[str] = Field(description="A list of the specific concepts, formulas, diseases, or structures tested in this document. Keep them short (e.g., 'Glomerular Filtration Rate', 'Class 2 Levers').")
    test_traps: list[str] = Field(description="A list of tricky wordings, unit conversions, or common mistakes found in these questions.")

structured_llm = llm.with_structured_output(ExtractedConcepts)

cruncher_prompt = SystemMessage(content="""You are a data-mining AI. 
Read the provided Science Olympiad test questions. Extract highly specific event topics being tested (e.g., specific household toxins, poisonous plants/animals, spill dynamics, exact chemical reactions). 
DO NOT extract basic, foundational science concepts like 'Atomic Structure', 'VSEPR', or 'Basic Stoichiometry'. Only extract the specific advanced applications and any obvious traps or tricky unit conversions. 
Standardize the names of the concepts.""")

# ==========================================
# 2. CHECKPOINT LOADING LOGIC
# ==========================================
TESTS_FOLDER = "raw_tests"
PROGRESS_FILE = "cruncher_save_state.json"

if not os.path.exists(TESTS_FOLDER):
    os.makedirs(TESTS_FOLDER)
    print(f"Created '{TESTS_FOLDER}' folder. Drop your .pdf or .docx test files in there and run again.")
    sys.exit()

test_files = [f for f in os.listdir(TESTS_FOLDER) if f.endswith('.pdf') or f.endswith('.docx')]

if not test_files:
    print(f"No .pdf or .docx files found in '{TESTS_FOLDER}'. Add your tests!")
    sys.exit()

# Initialize our master lists
master_concept_list = []
master_trap_list = []
processed_files = []

# Load previous progress if it exists
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

# ==========================================
# 3. THE TEST PROCESSING LOOP
# ==========================================
for filename in test_files:
    # SKIP files we already processed yesterday!
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
        result = structured_llm.invoke(messages)
        
        master_concept_list.extend(result.core_concepts)
        master_trap_list.extend(result.test_traps)
        processed_files.append(filename)
        
        # 🔥 SAVE STATE AFTER EVERY SUCCESSFUL FILE 🔥
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "concepts": master_concept_list,
                "traps": master_trap_list,
                "processed_files": processed_files
            }, f, indent=4)
            
    except Exception as e:
        error_msg = str(e).lower()
        # Look for rate limit keywords in the error message
        if "429" in error_msg or "quota" in error_msg or "exhausted" in error_msg:
            print(f"\n🚨 API LIMIT HIT (429)! Google is cutting us off for today.")
            print(f"Don't worry, your progress is safely saved. Just run this exact script again tomorrow!")
            sys.exit(0) # Stop the script cleanly
        else:
            print(f"   ❌ Failed to parse {filename}: {e}")

# ==========================================
# 4. BUILD THE FINAL LEADERBOARD
# ==========================================
# This only runs if the loop finishes completely without hitting a 429
print("\nAll files processed! Tallying final frequencies...")
concept_counts = Counter(master_concept_list)
trap_counts = Counter(master_trap_list)

frequency_report = {
    "Top_50_Tested_Concepts": [f"{concept} (Tested {count} times)" for concept, count in concept_counts.most_common(50)],
    "Top_20_Test_Traps": [f"{trap} (Seen {count} times)" for trap, count in trap_counts.most_common(20)]
}

with open("test_frequency_map.json", "w", encoding="utf-8") as f:
    json.dump(frequency_report, f, indent=4)

print("✅ SUCCESS! Master frequency leaderboard saved to 'test_frequency_map.json'.")