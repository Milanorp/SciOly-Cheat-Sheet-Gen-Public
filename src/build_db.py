import os
import time
import re # <-- NEW IMPORT FOR FIXING TEXT
from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# 0. Load your Google API Key from the .env file
load_dotenv()

# 1. Read the Markdown file
print("1. Reading extracted_rules.md...")
with open("extracted_rules.md", "r", encoding="utf-8") as f:
    text = f.read()

# --- THE HEADER FIX ---
print("2. Pre-processing the Markdown to fix Header Collisions...")
# This automatically changes "## 1." to "### 1." so it doesn't overwrite the Event Name!
text = re.sub(r'## (\d+\.)', r'### \1', text)

# 3. Tagging sections with Metadata based on Markdown Headers
print("3. Tagging sections with Metadata...")
headers_to_split_on = [
    ("##", "Event"), 
    ("###", "Section") # Now we capture the subsections safely!
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
print("6. Building database in batches to respect free tier limits...")

vectorstore = Chroma(
    persist_directory="./scioly_db", 
    embedding_function=embeddings
)

batch_size = 80 
total_chunks = len(chunked_documents)

for i in range(0, total_chunks, batch_size):
    batch = chunked_documents[i : i + batch_size]
    print(f"Processing chunks {i + 1} to {min(i + batch_size, total_chunks)} out of {total_chunks}...")
    
    vectorstore.add_documents(documents=batch)
    
    if i + batch_size < total_chunks:
        print("Pausing for 65 seconds to reset the API speed limit. Do not close the terminal...")
        time.sleep(65)

print("\nDatabase fully built, tagged, and saved to the './scioly_db' folder!")

# ---------------------------------------------------------
# 7. THE TEST: Let's search the database!
# ---------------------------------------------------------
print("\n--- Testing the Google Database ---")
query = "What is the penalty for adjusting a Scrambler?" 

# Let's test the Scrambler event specifically
results = vectorstore.as_retriever(search_kwargs={"k": 1, "filter": {"Event": "Scrambler"}}).invoke(query)

for res in results:
    print(f"\n[Attached Digital Tags]: {res.metadata}") 
    print(f"--- Top Result ---\n{res.page_content}")