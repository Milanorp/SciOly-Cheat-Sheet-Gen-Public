import json
import os
import re
import time
from langchain_core.messages import HumanMessage, SystemMessage
from src.token_tracker import TokenTrackerCallback
from src.factory import factory

TYPST_MATH_KEYWORDS = {
    'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta', 'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi', 'omicron', 'pi', 'rho', 'sigma', 'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega',
    'Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta', 'Theta', 'Iota', 'Kappa', 'Lambda', 'Mu', 'Nu', 'Xi', 'Omicron', 'Pi', 'Rho', 'Sigma', 'Tau', 'Upsilon', 'Phi', 'Chi', 'Psi', 'Omega',
    'sin', 'cos', 'tan', 'csc', 'sec', 'cot', 'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh', 'log', 'ln', 'lg', 'exp', 'max', 'min', 'lim', 'sup', 'inf', 'det', 'mod', 'Pr',
    'in', 'ni', 'subset', 'supset', 'times', 'div', 'approx', 'prop', 'ell', 'degree', 'dot', 'sum', 'prod', 'int', 'oint'
}

def quote_math_words(text: str) -> str:
    parts = re.split(r'("[^"]*")', text)
    for i in range(len(parts)):
        if i % 2 == 0:
            parts[i] = re.sub(r'[a-zA-Z]{2,}', lambda m: m.group(0) if m.group(0) in TYPST_MATH_KEYWORDS else f'"{m.group(0)}"', parts[i])
    return "".join(parts)

def parse_and_escape_typst(raw_text: str) -> str:
    """
    Parses tagged Markdown into Typst syntax.
    Splits text by [MATH]...[/MATH] and [CHEM]...[/CHEM] tags.
    """
    # Normalize all tags to $ to handle LLM hallucinations (e.g. mixed $ and [/MATH])
    raw_text = raw_text.replace('[MATH]', '$').replace('[/MATH]', '$')
    raw_text = raw_text.replace('[CHEM]', '$').replace('[/CHEM]', '$')
    
    # Now all math is wrapped in $...$
    pattern = re.compile(r'(\$.*?\$)', flags=re.DOTALL)
    parts = pattern.split(raw_text)
    
    escaped_parts = []
    
    for part in parts:
        if part.startswith('$') and part.endswith('$') and len(part) >= 2:
            content = part[1:-1].strip()
            content = re.sub(r'\s+', ' ', content)
            content = re.sub(r'\\text\{([^}]*)\}', r'"\1"', content) # convert \text{...} safely
            content = content.replace('{', '(').replace('}', ')') # translate LaTeX {} grouping to Typst () grouping
            content = content.replace('\\', '') # \alpha -> alpha
            content = content.replace('!!!!/', '') # Fix W\!\!\!\!/ hack
            
            # Translate LaTeXisms to Typst
            content = re.sub(r'\bcirc\b', 'degree', content)
            content = re.sub(r'\bpropto\b', 'prop', content)
            content = re.sub(r'\brightarrow\b', '->', content)
            content = re.sub(r'\bleftarrow\b', '<-', content)
            content = re.sub(r'\bleftrightarrow\b', '<->', content)
            content = re.sub(r'\bRightarrow\b', '=>', content)
            content = re.sub(r'\bLeftarrow\b', '<=', content)
            content = re.sub(r'\bLeftrightarrow\b', '<=>', content)
            content = re.sub(r'\bcdot\b', 'dot', content)

            content = re.sub(r'/\s*$', '', content) # Remove trailing slashes
            content = re.sub(r'_\s*$', '', content) # Remove trailing underscores
            content = re.sub(r'\^\s*$', '', content) # Remove trailing carets
            content = re.sub(r'^\s*\^', r'""^', content) # Fix leading carets
            content = re.sub(r'^\s*_', r'""_', content) # Fix leading underscores
            content = quote_math_words(content)
            escaped_parts.append(f"${content}$")
        else:
            text = part
            # Escape Typst special characters that are not in math
            text = text.replace('$', r'\$')
            text = text.replace('#', r'\#')
            
            # Convert Markdown to Typst formatting
            text = re.sub(r'\*\*([^\*]+)\*\*', r'*\1*', text)
            text = re.sub(r'(?<!\*)\*([^\*]+)\*(?!\*)', r'_\1_', text)
            
            escaped_parts.append(text)
            
    return "".join(escaped_parts)

def run(notes: dict) -> str:
    config = factory.get_config()
    OUTPUT_FILE = config['paths']['output_typ']

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
    print("PHASE 3: THE AI CHEAT SHEET COMPILER (TYPST RECOVERY)")
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
        5. EXPLICIT TAGS FOR MATH: For ALL mathematical formulas, units, variables, and scientific notation, wrap them EXACTLY in [MATH] ... [/MATH] tags. IMPORTANT: Use Typst math syntax inside these tags, NOT LaTeX. Do not use backslashes. For example, write [MATH]1.5 times 10^3[/MATH] or [MATH]"LD"_50[/MATH] or [MATH]alpha + beta[/MATH]. Do not use $...$.
        6. EXPLICIT TAGS FOR CHEMISTRY: For ALL chemical formulas and equations, wrap them EXACTLY in [CHEM] ... [/CHEM] tags. Use Typst math syntax. Example: [CHEM]H_2 O[/CHEM] or [CHEM]Ca^(2+)[/CHEM].
        7. NO MANUAL ESCAPING: Do not attempt to manually escape characters like %, &, or _ outside of your tagged blocks. The system will handle all escaping automatically. Write naturally!
        8. NO EMOJIS or FILLER.""")
        
        synthesis_req = HumanMessage(content=f"Synthesize these disjointed notes into a cohesive block of Typst/Markdown-ready text:\n\n{raw_section_text}")
        
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
            
            # Run the parser to escape text and build the Typst blocks
            synthesized_content = parse_and_escape_typst(raw_content)

            # Convert tall inline fractions into compact inline fractions
            synthesized_content = synthesized_content.replace(r"\frac", "/")

            # Compact common display-style operators
            synthesized_content = synthesized_content.replace(r"\displaystyle", "")
            
            # Wrap section in a small bold header
            clean_section_name_escaped = clean_section_name.replace('#', r'\#')
            section_typst = (
                f"*_{clean_section_name_escaped}:_* "
                f"{synthesized_content} "
            )
            combined_content += section_typst
            
            total_words += len(synthesized_content.split())
            print("   ✅ Section compiled and Typst-formatted.")
            
        except Exception as e:
            print(f"   ❌ Error synthesizing section: {e}")
            combined_content += f"*_{clean_section_name}:_* Error synthesizing this section. "

    # BUILD THE FULL TYPST DOCUMENT
    columns = 3 if not single_column else 1
    
    # Clean margins string like '0.2in'
    margin_val = margins.strip()
    
    # Map LaTeX paper size to Typst paper size
    typst_paper = "us-letter" if "letter" in paper_size.lower() else "a4"

    typst_template = f"""
#set page(
  paper: "{typst_paper}",
  margin: {margin_val},
  columns: {columns}
)

#set text(
  font: "Inter",
  size: {content_font_size},
  top-edge: "x-height",
  bottom-edge: "baseline"
)

#set par(
  justify: true,
  leading: 0.15em,
  spacing: 0.15em
)

#show math.equation: set text(size: {content_font_size})
"""
    typst_template += combined_content

    try:
        final_typst = typst_template.strip()
        
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(final_typst)
        print(f"\n✅ SUCCESS! Final synthesized Typst cheat sheet saved to '{OUTPUT_FILE}'!")
        print(f"Approximate Word Count: {total_words} words.")
    except Exception as e:
         print(f"❌ Error saving final Typst file: {e}")
         final_typst = typst_template.strip()
         
    return final_typst

if __name__ == "__main__":
    notes_path = os.path.join(factory.get_config()['paths']['data_dir'], "raw_research_notes.json")
    with open(notes_path, 'r', encoding='utf-8') as f:
        notes = json.load(f)
    run(notes)
