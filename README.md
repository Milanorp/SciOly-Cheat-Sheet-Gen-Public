# Science Olympiad Autonomous Cheat Sheet Generator

An advanced, AI-powered toolkit designed to help Science Olympiad competitors automatically generate hyper-optimized, incredibly dense, rulebook-accurate cheat sheets. 

This project uses a multi-agent pipeline powered by **Gemini 2.5 Pro**, **LangChain**, and **LangGraph** to extract concepts from past tests, research them across official rules and academic databases (ArXiv), audit them for technical accuracy, and typeset them into a professional LaTeX document.

## 🚀 The 6-Phase Pipeline

1.  **Phase 0: The Test Cruncher** - Scans past tests (`.pdf`, `.docx`) in `raw_tests/` to identify high-yield concepts and common traps.
2.  **Phase 1: The Architect** - Cross-references test data with official rules to plan a 50-target blueprint.
3.  **Phase 1.5: Setup Cache** - Uploads rules to Gemini Context Caching to reduce token costs by ~90%.
4.  **Phase 2: The Research Dispatcher** - Runs a parallel LangGraph agent armed with ArXiv and Web tools to draft research notes.
5.  **Phase 2.5: The Research Auditor** - **(New)** A senior scientist agent audits drafts for accuracy, forcing re-research on low-quality topics.
6.  **Phase 3: The AI Compiler** - Synthesizes notes into professional LaTeX source code.
7.  **Phase 4: The Formatter** - Compiles a math-perfect, 1-column, high-density PDF.

## 🛠️ Setup & Installation

1.  **Clone the Repo:**
    ```bash
    git clone https://github.com/Milanorp/SciOly-Cheat-Sheet-Gen-Public.git
    cd SciOly-Cheat-Sheet-Gen-Public
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment:**
    Create a `.env` file in the root directory and add your Google API Key:
    ```env
    GOOGLE_API_KEY=your_api_key_here
    ```

## 📚 Workflow Guide

### 1. Build the Intelligent RAG Database
- Place your official Science Olympiad rulebook PDF in the root directory and name it `sample_rules.pdf`.
- Run `python src/extract.py` to convert it to Markdown.
- Run `python src/build_db.py`. This uses **Semantic Chunking** and AI-generated summary indexing to build a high-intelligence vector database in `scioly_db/`.

### 2. Prepare Past Tests
- Drop past test PDFs or DOCX files for your event into the `raw_tests/` folder.

### 3. Generate the Cheat Sheet
- Run the main orchestrator:
    ```bash
    python main.py
    ```
- Enter your event name (e.g., "Chemistry Lab", "Astronomy") when prompted.
- The pipeline will handle the rest, automatically re-researching any topics that fail the audit.

## ⚛️ LaTeX Rendering
The program generates a professional `.tex` file and attempts to compile it to PDF locally.
- **If you have a LaTeX compiler installed:** You will get `Final_Cheat_Sheet.pdf` immediately.
- **If not:** Simply upload `Final_Cheat_Sheet.tex` to **Overleaf.com** and hit 'Recompile' for a math-perfect result.

---
*Powered by Gemini 2.5 Pro.*
