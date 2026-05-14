import json
import os
import time
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from src.token_tracker import TokenTrackerCallback

def run(notes: dict) -> str:
    print("\n" + "="*60)
    print("PHASE 3: THE AI CHEAT SHEET COMPILER (NATIVE LATEX)")
    print("="*60)

    load_dotenv()
    tracker = TokenTrackerCallback(script_name="3_cheat_sheet_compiler")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3, callbacks=[tracker])

    OUTPUT_FILE = "Final_Cheat_Sheet.tex"

    if not notes:
        print("❌ Error: No notes provided. Please run Phase 2 first.")
        return ""

    combined_content = ""
    total_words = 0

    for section_name, items in notes.items():
        clean_section_name = section_name.replace("_", " ").strip().upper()
        print(f"Synthesizing Section: {clean_section_name}...")
        
        raw_section_text = ""
        for item in items:
            raw_section_text += f"{item['content']} "
            
        synthesizer_prompt = SystemMessage(content=r"""You are an expert Science Olympiad LaTeX Editor.
        Synthesize the provided notes into a single section formatted for a professional LaTeX document.
        
        STRICT RULES:
        1. RETAIN ALL FACTS: Do not lose any formulas, numbers, or key terms.
        2. PURE CONTINUOUS PROSE: Write the entire synthesis as one single, massive, continuous block of text. ABSOLUTELY NO line breaks or new paragraphs.
        3. PERFECT GRAMMAR & PUNCTUATION: The raw notes contain fragmented bullet points. You MUST convert these fragments into clear, concise, and complete English sentences. Every distinct fact must end with a period or semicolon.
        4. LATEX MATH: You MUST use proper LaTeX math environments ($...$ for inline, $$...$$ for block). 
        5. ESCAPE SPECIAL CHARACTERS: Ensure that special LaTeX characters like %, &, _, # are properly escaped (e.g., \% instead of %) UNLESS they are inside a math environment.
        6. NO FORMATTING: Do not use bold or italics inside the text block. Use plain text only.
        7. NO EMOJIS or FILLER.""")
        
        synthesis_req = HumanMessage(content=f"Synthesize these disjointed notes into a cohesive block of LaTeX-ready text:\n\n{raw_section_text}")
        
        try:
            synthesized_content = llm.invoke([synthesizer_prompt, synthesis_req]).content
            
            # Programmatic failsafe: strip newlines
            synthesized_content = synthesized_content.replace("\n", " ").strip()
            
            # Wrap section in a small bold header
            section_latex = f"\\noindent \\textbf{{{clean_section_name}}}: {synthesized_content}\n\n"
            combined_content += section_latex
            
            total_words += len(synthesized_content.split())
            print("   ✅ Section compiled and LaTeX-formatted.")
            
        except Exception as e:
            print(f"   ❌ Error synthesizing section: {e}")
            combined_content += f"\\noindent \\textbf{{{clean_section_name}}}: Error synthesizing this section.\n\n"

    # BUILD THE FULL LATEX DOCUMENT
    latex_template = r"""
\documentclass[9pt]{extarticle}
\usepackage[utf8]{inputenc}
\usepackage[margin=0.15in, letterpaper]{geometry}
\usepackage{amsmath, amssymb, amsfonts}
\usepackage{microtype} % Better kerning and density

\pagestyle{empty}
\setlength{\parindent}{0pt}
\setlength{\parskip}{2pt}
\renewcommand{\baselinestretch}{1.0}

\begin{document}
\fontsize{7pt}{8pt}\selectfont
"""
    latex_template += combined_content
    latex_template += "\n\\end{document}"

    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(latex_template.strip())
        print(f"\n✅ SUCCESS! Final synthesized LaTeX cheat sheet saved to '{OUTPUT_FILE}'!")
        print(f"Approximate Word Count: {total_words} words.")
    except Exception as e:
         print(f"❌ Error saving final LaTeX file: {e}")
         
    return latex_template.strip()

if __name__ == "__main__":
    run({})
