import json
import os
import re
import time
from langchain_core.messages import HumanMessage, SystemMessage
from src.token_tracker import TokenTrackerCallback
from src.factory import factory

def parse_and_escape_latex(raw_text: str) -> str:
    """
    Parses tagged Markdown into perfectly escaped LaTeX.
    Splits text by [MATH]...[/MATH] and [CHEM]...[/CHEM] tags.
    Escapes plain text, converts markdown, and rebuilds the string.
    """
    pattern = re.compile(r'(\[MATH\].*?\[/MATH\]|\[CHEM\].*?\[/CHEM\])', flags=re.DOTALL)
    parts = pattern.split(raw_text)
    
    escaped_parts = []
    
    for part in parts:
        if part.startswith('[MATH]') and part.endswith('[/MATH]'):
            content = part[6:-7].strip()
            content = re.sub(r'\s+', ' ', content)
            escaped_parts.append(f"${content}$")
        elif part.startswith('[CHEM]') and part.endswith('[/CHEM]'):
            content = part[6:-7].strip()
            content = re.sub(r'\s+', ' ', content)
            escaped_parts.append(f"$\\ce{{{content}}}$")
        else:
            text = part.replace('\\', '\\textbackslash{}')
            text = text.replace('{', '\\{').replace('}', '\\}')
            text = text.replace('%', '\\%').replace('$', '\\$')
            text = text.replace('&', '\\&').replace('#', '\\#')
            text = text.replace('_', '\\_').replace('^', '\\textasciicircum{}')
            text = text.replace('~', '\\textasciitilde{}')
            
            text = re.sub(r'\*\*([^\*]+)\*\*', r'\\textbf{\1}', text)
            text = re.sub(r'(?<!\*)\*([^\*]+)\*(?!\*)', r'\\textit{\1}', text)
            
            escaped_parts.append(text)
            
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
            
        synthesizer_prompt = SystemMessage(content=r"""You are an expert Science Olympiad Notes Synthesizer.
        Synthesize the provided notes into a single section of text.
        
        CRITICAL FORMATTING RULES:
        1. RETAIN ALL FACTS: Do not lose any formulas, numbers, or key terms.
        2. PURE CONTINUOUS PROSE: Write the entire synthesis as one single, massive, continuous block of text. ABSOLUTELY NO line breaks (\n), paragraph breaks, or carriage returns.
        3. PERFECT GRAMMAR & PUNCTUATION: Every distinct fact must end with a period or semicolon.
        4. USE MARKDOWN FOR TEXT: Write the text entirely in Markdown. If you need italics (e.g., for species names), use *italic*. Do NOT use any LaTeX commands (like \textit) for text.
        5. EXPLICIT TAGS FOR MATH: For ALL mathematical formulas, units, variables, and scientific notation, wrap them EXACTLY in [MATH] ... [/MATH] tags. Do not use $...$ or $$...$$. Example: [MATH]1.5 \times 10^3[/MATH] or [MATH]\text{LD}_{50}[/MATH].
        6. EXPLICIT TAGS FOR CHEMISTRY: For ALL chemical formulas and equations, wrap them EXACTLY in [CHEM] ... [/CHEM] tags. Do not use \ce{...}. Example: [CHEM]H2O[/CHEM] or [CHEM]Ca^2+[/CHEM].
        7. NO MANUAL ESCAPING: Do not attempt to manually escape characters like %, &, or _ outside of your tagged blocks. The system will handle all escaping automatically. Write naturally!
        8. NO EMOJIS or FILLER.""")
        
        synthesis_req = HumanMessage(content=f"Synthesize these disjointed notes into a cohesive block of LaTeX-ready text:\n\n{raw_section_text}")
        
        try:
            raw_content = llm.invoke([synthesizer_prompt, synthesis_req]).content
            
            if isinstance(raw_content, list):
                final_content = ""
                for item in raw_content:
                    if isinstance(item, dict) and "text" in item:
                        final_content += item["text"]
                    elif isinstance(item, str):
                        final_content += item
                raw_content = final_content
            else:
                raw_content = str(raw_content)
                
            # Collapse whitespace before parsing to ensure we don't break tags
            raw_content = re.sub(r'\s+', ' ', raw_content).strip()
            
            # Run the parser to escape text and build the LaTeX blocks
            synthesized_content = parse_and_escape_latex(raw_content)

            # Convert tall inline fractions into compact inline fractions
            synthesized_content = synthesized_content.replace(r"\frac", r"\tfrac")

            # Compact common display-style operators
            synthesized_content = synthesized_content.replace(r"\displaystyle", "")
            
            # Wrap section in a small bold header, use space instead of double newline
            # OPTIMIZED HEADER: prevents headers from inflating line height
            clean_section_name_escaped = clean_section_name.replace('&', r'\&').replace('%', r'\%')
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
\usepackage{{lmodern}}

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
        final_latex = latex_template.strip()
        
        # We no longer need to clean up double escapes or unescaped superscripts 
        # since parse_and_escape_latex handles this deterministically.
        
        # Pull math/Greek symbols out of \text{...} blocks which cause silent PDF truncation in nonstopmode
        for sym in ["mu", "theta", "sigma", "alpha", "beta", "delta", "Delta", "lambda", "epsilon", "pm", "approx", "ge", "le"]:
            final_latex = re.sub(rf'\\text\{{([^}}]*)\\{sym}([^}}]*)\}}', rf'\\{sym}\\text{{\1\2}}', final_latex)
        
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(final_latex)
        print(f"\n✅ SUCCESS! Final synthesized LaTeX cheat sheet saved to '{OUTPUT_FILE}'!")
        print(f"Approximate Word Count: {total_words} words.")
    except Exception as e:
         print(f"❌ Error saving final LaTeX file: {e}")
         final_latex = latex_template.strip()
         
    return final_latex

if __name__ == "__main__":
    notes_path = os.path.join(factory.get_config()['paths']['data_dir'], "raw_research_notes.json")
    with open(notes_path, 'r', encoding='utf-8') as f:
        notes = json.load(f)
    run(notes)
