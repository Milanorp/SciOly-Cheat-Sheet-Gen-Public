import os
import sys
import re
import json
import asyncio
import tenacity

# Add the project root to sys.path to allow 'from src.X import Y' imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from src.factory import factory, console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, MofNCompleteColumn

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

async def main():
    # 0. Load Factory Config
    config = factory.get_config()
    db_path = os.path.abspath(config['database']['db_path'])
    embeddings = factory.get_embeddings()
    llm = factory.get_llm(purpose="researcher")
    
    console.print("\n[phase]🚀 STARTING PARALLEL DATABASE BUILD 🚀[/phase]")

    all_docs_to_index = []

    # --- 1. INDEX THE RULES ---
    console.print("[info]1. Processing extracted_rules.md...[/info]")
    try:
        with open("extracted_rules.md", "r", encoding="utf-8") as f:
            rules_text = f.read()
            
        rules_text = re.sub(r'## (\d+\.)', r'### \1', rules_text)
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("##", "Event"), ("###", "Section")])
        md_docs = markdown_splitter.split_text(rules_text)

        for doc in md_docs:
            doc.metadata["Source_Type"] = "Rulebook"
            if "Event" in doc.metadata:
                doc.metadata["Event"] = doc.metadata["Event"].replace("**", "").replace(" B", "").replace("(CONT.)", "").strip().title()
        
        all_docs_to_index.extend(md_docs)
        console.print(f"   ✅ Rules processed: {len(md_docs)} segments.")
    except FileNotFoundError:
        console.print("[warning]⚠️ 'extracted_rules.md' not found. Skipping rule indexing.[/warning]")

    # --- 2. INDEX THE PAST TESTS ---
    TESTS_FOLDER = config['paths']['raw_tests']
    console.print(f"[info]2. Processing past tests from '{TESTS_FOLDER}'...[/info]")
    
    if os.path.exists(TESTS_FOLDER):
        test_files = [f for f in os.listdir(TESTS_FOLDER) if f.endswith('.pdf') or f.endswith('.docx')]
        for filename in test_files:
            filepath = os.path.join(TESTS_FOLDER, filename)
            try:
                if filename.endswith('.pdf'):
                    loader = PyPDFLoader(filepath)
                else:
                    loader = Docx2txtLoader(filepath)
                
                docs = loader.load()
                for doc in docs:
                    doc.metadata["Source_Type"] = "Past Test"
                    doc.metadata["Filename"] = filename
                    # Attempt to guess event from filename
                    doc.metadata["Event"] = filename.split('_')[0].replace(".pdf", "").replace(".docx", "").title()
                
                all_docs_to_index.extend(docs)
                console.print(f"   ✅ Indexed: {filename}")
            except Exception as e:
                console.print(f"[error]   ❌ Failed to parse {filename}: {e}[/error]")
    
    if not all_docs_to_index:
        console.print("[error]❌ No documents found to index. Pipeline stopped.[/error]")
        return

    # --- 3. SEMANTIC CHUNKING ---
    console.print(f"\n[info]3. Performing Semantic Chunking on {len(all_docs_to_index)} segments...[/info]")
    text_splitter = SemanticChunker(embeddings)
    chunked_documents = text_splitter.split_documents(all_docs_to_index)
    console.print(f"   ✅ Split into [success]{len(chunked_documents)}[/success] semantically coherent chunks.")

    # --- 4. METADATA ENRICHMENT ---
    console.print("\n[phase]4. Enriching chunks with AI Summary Tags...[/phase]")
    CONCURRENCY_LIMIT = config['research']['concurrency_limit']
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async def enrich_chunk(doc, progress, task_id):
        async with semaphore:
            try:
                @factory.get_retry_decorator()
                async def ainvoke_with_retry(prompt):
                    return await llm.ainvoke(prompt)

                src_type = doc.metadata.get("Source_Type", "Unknown")
                prompt = f"Summarize this Science Olympiad {src_type} chunk in one short, descriptive sentence for indexing. Content: {doc.page_content[:1500]}"
                summary_res = await ainvoke_with_retry(prompt)
                doc.metadata["Summary"] = summary_res.content.strip()
                progress.update(task_id, advance=1)
            except Exception as e:
                progress.update(task_id, advance=1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        console=console
    ) as progress:
        enrich_task = progress.add_task("[cyan]Enriching...", total=len(chunked_documents))
        tasks = [enrich_chunk(doc, progress, enrich_task) for doc in chunked_documents]
        await asyncio.gather(*tasks)

    # --- 5. WRITE TO DB ---
    console.print("\n[phase]5. Writing to Vector Database (ChromaDB)[/phase]")
    # Clear old DB if it exists to prevent duplication
    if os.path.exists(db_path):
        import shutil
        # shutil.rmtree(db_path) # Risky, let's just append or inform
        console.print("[info]Appending to existing database...[/info]")

    vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
    batch_size = 50 
    for i in range(0, len(chunked_documents), batch_size):
        batch = chunked_documents[i : i + batch_size]
        vectorstore.add_documents(documents=batch)

    console.print(f"\n[success]✅ Database updated and saved to '{db_path}'[/success]")

    # --- THE TEST ---
    console.print("\n[info]--- Testing Search Intelligence ---[/info]")
    query = "What is the penalty for adjusting a Scrambler?" 
    results = vectorstore.as_retriever(search_kwargs={"k": 1}).invoke(query)

    for res in results:
        console.print(f"\n[bold white]AI Summary Tag:[/bold white] [success]{res.metadata.get('Summary', 'N/A')}[/success]")
        console.print(f"[bold cyan]--- Top Result ---[/bold cyan]\n{res.page_content}")

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
