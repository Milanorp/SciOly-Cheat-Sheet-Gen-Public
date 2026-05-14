import os
import re
import json
import tenacity
from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
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

# 5. Connect to Google's cloud embedding model
print("4. Connecting to Google Cloud for Semantic Intelligence...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)

# 4. Chop the tagged text into manageable chunks using Semantic Intelligence
print("5. Chopping tagged sections into semantically coherent chunks...")
text_splitter = SemanticChunker(embeddings)
chunked_documents = text_splitter.split_documents(md_docs)
print(f"Successfully split into {len(chunked_documents)} semantically tagged chunks.")

# --- METADATA ENRICHMENT LOOP ---
print("6. Enriching chunks with AI-generated Summary Tags for better indexing...")

def is_rate_limit(exception):
    err_str = str(exception).lower()
    return "429" in err_str or "quota" in err_str or "exhausted" in err_str

@tenacity.retry(
    wait=tenacity.wait_exponential(multiplier=2, min=4, max=65),
    stop=tenacity.stop_after_attempt(5),
    retry=tenacity.retry_if_exception(is_rate_limit),
    before_sleep=lambda retry_state: print(f"⚠️ Rate limit hit during enrichment. Retrying in {retry_state.next_action.sleep} seconds...")
)
def enrich_doc(doc):
    prompt = f"Summarize the following Science Olympiad rule chunk in one short, descriptive sentence for indexing. Focus on the core rule, penalty, or specification described: {doc.page_content}"
    summary = llm.invoke(prompt).content
    doc.metadata["Summary"] = summary.strip()

total_to_enrich = len(chunked_documents)
for i, doc in enumerate(chunked_documents):
    if (i + 1) % 10 == 0 or i == 0:
        print(f"   Enriching chunk {i+1} of {total_to_enrich}...")
    enrich_doc(doc)

# 6. Build and save the Vector Database IN BATCHES
print("\n7. Building database in batches with robust rate limiting (tenacity)...")

vectorstore = Chroma(
    persist_directory="./scioly_db", 
    embedding_function=embeddings
)

batch_size = 50 
total_chunks = len(chunked_documents)

@tenacity.retry(
    wait=tenacity.wait_exponential(multiplier=2, min=4, max=65),
    stop=tenacity.stop_after_attempt(5),
    retry=tenacity.retry_if_exception(is_rate_limit),
    before_sleep=lambda retry_state: print(f"⚠️ Rate limit hit during DB write. Retrying in {retry_state.next_action.sleep} seconds...")
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
print("\n--- Testing the Enhanced Database ---")
query = "What is the penalty for adjusting a Scrambler?" 

results = vectorstore.as_retriever(search_kwargs={"k": 1, "filter": {"Event": "Scrambler"}}).invoke(query)

for res in results:
    print(f"\n[AI Summary Tag]: {res.metadata.get('Summary', 'N/A')}")
    print(f"[Attached Digital Tags]: {res.metadata}") 
    print(f"--- Top Result ---\n{res.page_content}")
