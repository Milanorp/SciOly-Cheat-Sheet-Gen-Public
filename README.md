# 🏆 Science Olympiad Autonomous Cheat Sheet Generator

An advanced, AI-powered toolkit designed to help Science Olympiad competitors automatically generate hyper-optimized, incredibly dense, rulebook-accurate cheat sheets. 

This project uses **LangChain**, **LangGraph**, and **Google's Gemini 2.5 Flash** to perform data extraction from past tests, build a local Retrieval-Augmented Generation (RAG) database from the official rulebook, and synthesize the information into a printable, maximum-density format.

---

## ⚙️ Prerequisites

1. **Python 3.10+** installed.
2. **Microsoft Edge** (Required for the final automatic PDF generation step).
3. **Google Gemini API Key:** You can get one for free at [Google AI Studio](https://aistudio.google.com/app/apikey).

## 🚀 Setup Instructions

1. **Clone the repository and enter the directory:**
   ```bash
   git clone https://github.com/Milanorp/SciOly_Cheat_Sheet_Generator.git
   cd SciOly_Cheat_Sheet_Generator
   ```

2. **Set up a Virtual Environment:**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your API Key:**
   Create a file named `.env` in the root of the project and add your Gemini API key:
   ```env
   GOOGLE_API_KEY=your_api_key_here
   ```

---

## 🛠️ Step 1: Feed the AI Data

Before generating a cheat sheet, you need to provide the AI with the rulebook and past tests.

1. **The Rulebook (RAG Database):**
   - Place the official Science Olympiad rulebook PDF in the root directory and name it `sample_rules.pdf` (or update `extract.py` with your file name).
   - Run `python extract.py` to convert the PDF to a readable Markdown format (`extracted_rules.md`).
   - Run `python build_db.py` to chunk the rules and build your local Chroma Vector Database (`scioly_db/`).

2. **Past Tests (Frequency Analysis):**
   - Create a folder named `raw_tests/` in the root directory.
   - Drop as many past test PDFs for your event into this folder as you want.
   - Run `python 0_test_cruncher.py`. This script will read every test and generate a `test_frequency_map.json` leaderboard of the most commonly tested concepts and traps.

---

## 🏭 Step 2: The One-Click Generation Pipeline

Once the AI has the rules and the past test data, generating the ultimate cheat sheet is as easy as running a single command:

```bash
python main.py
```

The orchestrator script will automatically run the 5-phase generation pipeline in sequence and stream the output directly to your terminal. 

### What happens under the hood:

*   **Phase 1 (The Architect):** Reads the past test frequency map and strategically plans a 50-target layout designed to perfectly fill a standard 8.5x11 page (front and back).
*   **Phase 1.5 (Upload Rulebook Cache):** Uploads your extracted rulebook to Gemini's secure Context Caching server for 24 hours, drastically speeding up generation and reducing API token costs by 75%.
*   **Phase 2 (The Research Dispatcher):** The heavy lifter. It takes the 50 targets, expands them into detailed checklists, queries your local rulebook database AND the high-speed Cache, and drafts ~130 words of highly technical notes for each target. *(Note: If you hit a `429 Rate Limit` on the Free tier, the script automatically saves its state! Just run `python main.py` again to resume.)*
*   **Phase 3 (The AI Compiler):** Takes the disjointed research notes and synthesizes them into massive, continuous blocks of pure text, stripping out formatting to achieve "Extreme Density".
*   **Phase 4 (The PDF Formatter):** Injects custom Science Olympiad CSS (5.5pt font, zero margins), and automatically uses a headless Microsoft Edge browser to render and export the final `Final_Cheat_Sheet.pdf`.

---

## 🤖 Bonus: The Autonomous Graph Agent

If you just want to ask complex questions without generating a full cheat sheet, use the interactive terminal agent!

```bash
python graph_agent.py
```
This agent uses LangGraph to autonomously decide when to search your local rulebook database, when to search live academic papers on ArXiv, and when to search the web to answer complex build or testing questions.
