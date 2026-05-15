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

async def main():
    # 0. Load Factory Config
    config = factory.get_config()

    console.print("\n[phase]🚀 STARTING PARALLEL DATABASE BUILD 🚀[/phase]")

    # 1. Read the Markdown file
    console.print("[info]1. Reading extracted_rules.md...[/info]")
    try:
        with open("extracted_rules.md", "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        console.print("[error]❌ Error: 'extracted_rules.md' not found. Please extract the rules first.[/error]")
        return

    # --- THE HEADER FIX ---
    console.print("[info]2. Pre-processing Markdown headers...[/info]")
    text = re.sub(r'## (\d+\.)', r'### \1', text)

    # 3. Tagging sections with Metadata based on Markdown Headers
    console.print("[info]3. Tagging sections with Metadata...[/info]")
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

    console.print(f"Detected [success]{len(md_docs)}[/success] distinct event/section chunks.")

    # 5. Connect to Google's cloud embedding model
    console.print("[info]4. Connecting to Google Cloud for Semantic Intelligence...[/info]")
    embeddings = factory.get_embeddings()
    llm = factory.get_llm(purpose="researcher")

    # 4. Chop the tagged text into manageable chunks using Semantic Intelligence
    console.print("[info]5. Performing Semantic Chunking (Coherent Topic Detection)...[/info]")
    text_splitter = SemanticChunker(embeddings)
    chunked_documents = text_splitter.split_documents(md_docs)
    console.print(f"Successfully split into [success]{len(chunked_documents)}[/success] semantically tagged chunks.")

    # --- METADATA ENRICHMENT LOOP (PARALLEL) ---
    console.print("\n[phase]6. Enriching chunks with AI Summary Tags (Parallel Mode)[/phase]")

    CONCURRENCY_LIMIT = config['research']['concurrency_limit']
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async def enrich_chunk(doc, progress, task_id):
        async with semaphore:
            try:
                # Get the retry decorator and apply it to the ainvoke call
                @factory.get_retry_decorator(
                    before_sleep_func=lambda retry_state: console.print(f"[warning]⚠️ Rate limit hit. Retrying in {retry_state.next_action.sleep}s...[/warning]")
                )
                async def ainvoke_with_retry(prompt):
                    return await llm.ainvoke(prompt)

                prompt = f"Summarize the following Science Olympiad rule chunk in one short, descriptive sentence for indexing. Focus on the core rule, penalty, or specification described: {doc.page_content}"
                summary_res = await ainvoke_with_retry(prompt)
                doc.metadata["Summary"] = summary_res.content.strip()
                progress.update(task_id, advance=1)
            except Exception as e:
                console.print(f"[error]❌ Error enriching chunk: {e}[/error]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        console=console
    ) as progress:
        enrich_task = progress.add_task("[cyan]Enriching Chunks...", total=len(chunked_documents))
        tasks = [enrich_chunk(doc, progress, enrich_task) for doc in chunked_documents]
        await asyncio.gather(*tasks)

    # 6. Build and save the Vector Database IN BATCHES
    console.print("\n[phase]7. Writing to Vector Database (ChromaDB)[/phase]")

    db_path = os.path.abspath(config['database']['db_path'])
    vectorstore = Chroma(
        persist_directory=db_path, 
        embedding_function=embeddings
    )

    batch_size = 50 
    total_chunks = len(chunked_documents)

    @factory.get_retry_decorator(
        before_sleep_func=lambda retry_state: console.print(f"[warning]⚠️ DB Rate limit hit. Retrying in {retry_state.next_action.sleep}s...[/warning]")
    )
    def add_docs_with_retry(batch):
        vectorstore.add_documents(documents=batch)

    for i in range(0, total_chunks, batch_size):
        batch = chunked_documents[i : i + batch_size]
        console.print(f"   Writing chunks {i + 1} to {min(i + batch_size, total_chunks)}...")
        add_docs_with_retry(batch)

    console.print(f"\n[success]✅ Database fully built and saved to '{db_path}'[/success]")

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
