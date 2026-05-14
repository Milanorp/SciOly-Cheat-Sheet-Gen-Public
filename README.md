# Science Olympiad Autonomous Cheat Sheet Generator

An advanced, AI-powered toolkit designed to help Science Olympiad competitors automatically generate hyper-optimized, incredibly dense, rulebook-accurate cheat sheets. 

This project uses a professional-grade multi-agent pipeline powered by **Gemini 2.5 Pro**, **LangChain**, and **LangGraph**. It features a centralized architecture with YAML configuration, automated state management, and a native LaTeX typesetting engine.

## 🚀 The 6-Phase Pipeline

1.  **Phase 0: The Test Cruncher** - Scans past tests (`.pdf`, `.docx`) in `raw_tests/` to identify high-yield concepts and common traps.
2.  **Phase 1: The Architect** - Cross-references test data with official rules to plan a high-density blueprint.
3.  **Phase 1.5: Setup Cache** - Uploads rules to Gemini Context Caching to reduce token costs by ~90%.
4.  **Phase 2: The Research Dispatcher** - Runs a parallel LangGraph agent armed with ArXiv and Web tools to draft research notes.
5.  **Phase 2.5: The Research Auditor** - A senior scientist agent audits drafts for accuracy and technical depth, forcing re-research on failed topics.
6.  **Phase 3: The AI Compiler** - Synthesizes notes into professional, LaTeX-ready source code.
7.  **Phase 4: The Formatter** - Compiles a math-perfect, 1-column, high-density PDF.

## 🏗️ System Architecture

- **Centralized Config (`config/settings.yaml`):** Easily customize model types, target counts, word limits, and document formatting without touching code.
- **Global State Manager:** Robust checkpointing ensures the pipeline can resume from the exact target it stopped at.
- **Model Factory:** Unified management of LLM instantiation and `tenacity` retry logic for extreme reliability.
- **Type Safety:** Built with **Pydantic** models to ensure data integrity across all 6 phases.

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
- Run `python src/build_db.py`. This uses **Semantic Chunking** and AI-generated summary indexing to build a high-intelligence vector database.

### 2. Prepare Past Tests
- Drop past test PDFs or DOCX files for your event into the `raw_tests/` folder.

### 3. Generate the Cheat Sheet
- Run the main orchestrator:
    ```bash
    python main.py
    ```
- The system will load your configuration from `config/settings.yaml` and handle the entire research and audit loop autonomously.

## ⚛️ LaTeX Rendering
The program generates a professional `.tex` file and attempts to compile it to PDF locally.
- **If you have a LaTeX compiler (pdflatex, tectonic) installed:** You will get `Final_Cheat_Sheet.pdf` immediately.
- **If not:** Simply upload `Final_Cheat_Sheet.tex` to **Overleaf.com** and hit 'Recompile' for a math-perfect result.

---
*Powered by Gemini API.*
