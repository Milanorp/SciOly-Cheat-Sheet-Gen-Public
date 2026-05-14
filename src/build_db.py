import os
import re
import json
import tenacity
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from src.factory import factory

# 0. Load Factory Config
config = factory.get_config()

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
embeddings = factory.get_embeddings()
llm = factory.get_llm(purpose="researcher")

# 4. Chop the tagged text into manageable chunks using Semantic Intelligence
print("5. Chopping tagged sections into semantically coherent chunks...")
text_splitter = SemanticChunker(embeddings)
chunked_documents = text_splitter.split_documents(md_docs)
print(f"Successfully split into {len(chunked_documents)} semantically tagged chunks.")

# --- METADATA ENRICHMENT LOOP ---
print("6. Enriching chunks with AI-generated Summary Tags for better indexing...")

enrich_with_retry = factory.get_retry_decorator(
    before_sleep_func=lambda retry_state: print(f"⚠️ Rate limit hit during enrichment. Retrying in {retry_state.next_action.sleep} seconds...")
)(llm.invoke)

for i, doc in enumerate(chunked_documents):
    if (i + 1) % 10 == 0 or i == 0:
        print(f"   Enriching chunk {i+1} of {len(chunked_documents)}...")
    
    prompt = f"Summarize the following Science Olympiad rule chunk in one short, descriptive sentence for indexing. Focus on the core rule, penalty, or specification described: {doc.page_content}"
    summary = enrich_with_retry(prompt).content
    doc.metadata["Summary"] = summary.strip()

# 6. Build and save the Vector Database IN BATCHES
print("\n7. Building database in batches with robust rate limiting...")

vectorstore = Chroma(
    persist_directory=config['database']['db_path'], 
    embedding_function=embeddings
)

batch_size = 50 
total_chunks = len(chunked_documents)

add_docs_with_retry = factory.get_retry_decorator(
    before_sleep_func=lambda retry_state: print(f"⚠️ Rate limit hit during DB write. Retrying in {retry_state.next_action.sleep} seconds...")
)(vectorstore.add_documents)

for i in range(0, total_chunks, batch_size):
    batch = chunked_documents[i : i + batch_size]
    print(f"Processing chunks {i + 1} to {min(i + batch_size, total_chunks)} out of {total_chunks}...")
    
    add_docs_with_retry(batch)

print(f"\nDatabase fully built, tagged, and saved to the '{config['database']['db_path']}' folder!")

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
