import json
import os
import re
import time
from langchain_core.messages import HumanMessage, SystemMessage
from src.token_tracker import TokenTrackerCallback
from src.factory import factory

def latex_escape(text: str) -> str:
    """
    Escapes LaTeX special characters while preserving everything inside $...$ (inline math)
    and $$...$$ (block math).
    """
    # Pattern to find math environments
    # This matches $$...$$ or $...$
    math_pattern = r'(\$\$.*?\$\$|\$.*?\$)'
    
    # Split the text by math environments
    parts = re.split(math_pattern, text, flags=re.DOTALL)
    
    escaped_parts = []
    for i, part in enumerate(parts):
        # Even indices are non-math text, odd indices are math environments
        if i % 2 == 0:
            # Escape special LaTeX characters in non-math text
            # Order matters: backslash first
            part = part.replace('\\', r'\textbackslash{}')
            part = part.replace('&', r'\&')
            part = part.replace('%', r'\%')
            part = part.replace('$', r'\$')
            part = part.replace('#', r'\#')
            part = part.replace('_', r'\_')
            part = part.replace('{', r'\{')
            part = part.replace('}', r'\}')
            part = part.replace('~', r'\textasciitilde{}')
            part = part.replace('^', r'\textasciicircum{}')
            # Clean up the backslash placeholder
            part = part.replace(r'\textbackslash{}', r'\\') # Wait, no. \textbackslash is safer.
            # Actually, the AI uses \rightarrow and \Delta. 
            # If I escape backslashes, I break the AI's LaTeX commands.
            
            # REVISION: The AI is supposed to be outputting LaTeX. 
            # Let's ONLY escape the ones that are most likely to be typos in plain text: & and %
            # and only if they aren't obviously part of a command.
            # But the Auditor should have caught most.
            
            # Let's stick to a simpler set that covers 90% of crashes:
            part = part.replace('&', r'\&')
            part = part.replace('%', r'\%')
            # part = part.replace('_', r'\_') # Many AI's forget this one
            
        escaped_parts.append(part)
        
    return "".join(escaped_parts)

def run(notes: dict) -> str:
    config = factory.get_config()
    OUTPUT_FILE = config['paths']['output_tex']

    # PRE-LOAD FORMATTING (Required for header synthesis)
    font_size = config['formatting']['font_size']
    content_font_size = config['formatting']['content_font_size']
    line_height = config['formatting']['line_height']
    margins = config['formatting']['margins']
    paper_size = config['formatting']['paper_size']
    single_column = config['formatting'].get('single_column', True)

    # Clean font size for float calculations
    clean_fs = float(content_font_size.replace("pt", "").strip())

    print("\n" + "="*60)
    print("PHASE 3: THE AI CHEAT SHEET COMPILER (LATEX RECOVERY)")
    print("="*60)

    tracker = TokenTrackerCallback(script_name="3_cheat_sheet_compiler")
    llm = factory.get_llm(purpose="compiler")
    llm.callbacks = [tracker]

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
        
        CRITICAL DENSITY RULES:
        1. RETAIN ALL FACTS: Do not lose any formulas, numbers, or key terms.
        2. PURE CONTINUOUS PROSE: Write the entire synthesis as one single, massive, continuous block of text. ABSOLUTELY NO line breaks (\n), paragraph breaks, or carriage returns.
        3. PERFECT GRAMMAR & PUNCTUATION: Every distinct fact must end with a period or semicolon.
        4. LATEX MATH: Use $...$ for inline math. ABSOLUTELY FORBIDDEN: $$...$$, \begin{equation}, \begin{align}, or any display math environments. NEVER put a formula on its own line.
        5. CHEMISTRY: Use the \ce{...} command from the mhchem package for ALL chemical formulas (e.g., \ce{H2O}, \ce{CO2}, \ce{Na2SO4.10H2O}, \ce{Ca^2+}).
        6. UNIFORMITY: Ensure all units and scientific notation are in math mode (e.g., $1.5 \times 10^3$, $58^\circ\text{C}$, $\text{g/cm}^3$).
        7. ESCAPE SPECIAL CHARACTERS: Ensure that special LaTeX characters like %, &, _, # are properly escaped (e.g., \% instead of %) UNLESS they are inside a math environment ($...$) or \ce{...}.
        8. NO FORMATTING: Do not use bold or italics inside the text block. Use plain text only.
        9. NO EMOJIS or FILLER.""")
        
        synthesis_req = HumanMessage(content=f"Synthesize these disjointed notes into a cohesive block of LaTeX-ready text:\n\n{raw_section_text}")
        
        try:
            synthesized_content = llm.invoke([synthesizer_prompt, synthesis_req]).content
            
            # Collapse whitespace
            synthesized_content = re.sub(r'\s+', ' ', synthesized_content).strip()

            # Convert tall inline fractions into compact inline fractions
            synthesized_content = synthesized_content.replace(r"\frac", r"\tfrac")

            # Compact common display-style operators
            synthesized_content = synthesized_content.replace(r"\displaystyle", "")

            # Escape dangerous characters
            synthesized_content = synthesized_content.replace("&", r"\&").replace("%", r"\%")
            
            # Wrap section in a small bold header, use space instead of double newline
            # OPTIMIZED HEADER: prevents headers from inflating line height
            clean_section_name_escaped = clean_section_name.replace("&", r"\&").replace("%", r"\%")
            section_latex = (
                f"\\noindent "
                f"{{\\bfseries\\fontsize{{{max(clean_fs-0.5, 4.5)}}}"
                f"{{{max(clean_fs-0.5, 4.5)}}}\\selectfont "
                f"{clean_section_name_escaped}:}} "
                f"{synthesized_content} "
            )
            combined_content += section_latex
            
            total_words += len(synthesized_content.split())
            print("   ✅ Section compiled and LaTeX-formatted.")
            
        except Exception as e:
            print(f"   ❌ Error synthesizing section: {e}")
            combined_content += f"\\noindent \\textbf{{{clean_section_name}}}: Error synthesizing this section. "

    # BUILD THE FULL LATEX DOCUMENT
    column_start = ""
    column_end = ""
    if not single_column:
        column_start = "\\begin{multicols}{3}\n"
        column_end = "\\end{multicols}\n"

    # Use a safe template without f-string brace collisions for the static parts
    latex_template = fr"""
\documentclass[{font_size}]{{extarticle}}

\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}

\usepackage[
    margin={margins},
    {paper_size}
]{{geometry}}

\usepackage{{amsmath,amssymb,amsfonts}}
\usepackage[version=4]{{mhchem}}
\usepackage{{multicol}}
\usepackage{{microtype}}
\usepackage{{enumitem}}
\usepackage{{ragged2e}}

% ---------- PAGE STYLE ----------
\pagestyle{{empty}}
\raggedbottom

% ---------- PARAGRAPH CONTROL ----------
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{0pt}}

% ---------- LINE SPACING ----------
\renewcommand{{\baselinestretch}}{{0.92}}

% Prevent equations from expanding lines
\lineskip=0pt
\lineskiplimit=0pt

% ---------- INLINE MATH COMPRESSION ----------
\everymath{{\scriptstyle}}
\everydisplay{{\scriptstyle}}

% Compact display spacing
\abovedisplayskip=0pt
\belowdisplayskip=0pt
\abovedisplayshortskip=0pt
\belowdisplayshortskip=0pt

% Compact math spacing
\thinmuskip=1mu
\medmuskip=1mu plus 1mu minus 1mu
\thickmuskip=1mu plus 1mu minus 1mu

% ---------- CHEMISTRY COMPRESSION ----------
\mhchemoptions{{textfontcommand=\scriptsize}}

% ---------- MICROTYPOGRAPHY ----------
\microtypesetup{{
    protrusion=true,
    expansion=true
}}

% ---------- DOCUMENT ----------
\begin{{document}}

\fontsize{{{content_font_size}}}{{{clean_fs + 0.3}}}\selectfont

{column_start}
"""
    latex_template += combined_content

    latex_template += rf"""

\enlargethispage{{2\baselineskip}}

{column_end}

\end{{document}}
"""

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
