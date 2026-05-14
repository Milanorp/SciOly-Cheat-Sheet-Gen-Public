import os
import re
import tenacity
from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# 0. Load your Google API Key from the .env file
load_dotenv()

# 1. Read the Markdown file
print("1. Reading extracted_rules.md...")
try:
    with open("extracted_rules.md", "r", encoding="utf-8") as f:
        text = f.read()
except FileNotFoundError:
    print("❌ Error: 'extracted_rules.md' not found. Please extract the rules first.")
    exit(1)

# --- THE HEADER FIX ---
print("2. Pre-processing the Markdown to fix Header Collisions...")
text = re.sub(r'## (\d+\.)', r'### \1', text)

# 3. Tagging sections with Metadata based on Markdown Headers
print("3. Tagging sections with Metadata...")
headers_to_split_on = [
    ("##", "Event"), 
    ("###", "Section")
]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_docs = markdown_splitter.split_text(text)

# --- THE METADATA SCRUBBER ---
for doc in md_docs:
    if "Event" in doc.metadata:
        raw_tag = doc.metadata["Event"]
        clean_tag = raw_tag.replace("**", "").replace(" B", "").replace("(CONT.)", "").strip().title()
        doc.metadata["Event"] = clean_tag

print(f"Detected {len(md_docs)} distinct event/section chunks.")

# 4. Chop the tagged text into manageable chunks
print("4. Chopping tagged sections into smaller chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,   
    chunk_overlap=200  
)
chunked_documents = text_splitter.split_documents(md_docs)
print(f"Successfully split into {len(chunked_documents)} tagged chunks.")

# 5. Connect to Google's cloud embedding model
print("5. Connecting to Google Cloud...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# 6. Build and save the Vector Database IN BATCHES
print("6. Building database in batches with robust rate limiting (tenacity)...")

vectorstore = Chroma(
    persist_directory="./scioly_db", 
    embedding_function=embeddings
)

batch_size = 80 
total_chunks = len(chunked_documents)

def is_rate_limit(exception):
    err_str = str(exception).lower()
    return "429" in err_str or "quota" in err_str or "exhausted" in err_str

@tenacity.retry(
    wait=tenacity.wait_exponential(multiplier=2, min=4, max=65),
    stop=tenacity.stop_after_attempt(5),
    retry=tenacity.retry_if_exception(is_rate_limit),
    before_sleep=lambda retry_state: print(f"⚠️ Rate limit hit. Retrying in {retry_state.next_action.sleep} seconds...")
)
def add_documents_with_retry(batch):
    vectorstore.add_documents(documents=batch)

for i in range(0, total_chunks, batch_size):
    batch = chunked_documents[i : i + batch_size]
    print(f"Processing chunks {i + 1} to {min(i + batch_size, total_chunks)} out of {total_chunks}...")
    
    add_documents_with_retry(batch)

print("\nDatabase fully built, tagged, and saved to the './scioly_db' folder!")

# ---------------------------------------------------------
# 7. THE TEST: Let's search the database!
# ---------------------------------------------------------
print("\n--- Testing the Google Database ---")
query = "What is the penalty for adjusting a Scrambler?" 

results = vectorstore.as_retriever(search_kwargs={"k": 1, "filter": {"Event": "Scrambler"}}).invoke(query)

for res in results:
    print(f"\n[Attached Digital Tags]: {res.metadata}") 
    print(f"--- Top Result ---\n{res.page_content}")
