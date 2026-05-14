import json
import os
import time
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from src.token_tracker import TokenTrackerCallback

def run(notes: dict) -> str:
    print("\n" + "="*60)
    print("PHASE 3: THE AI CHEAT SHEET COMPILER & SYNTHESIZER")
    print("="*60)

    load_dotenv()
    tracker = TokenTrackerCallback(script_name="3_cheat_sheet_compiler")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, callbacks=[tracker])

    OUTPUT_FILE = "Final_Cheat_Sheet.md"

    if not notes:
        print("❌ Error: No notes provided. Please run Phase 2 first.")
        return ""

    markdown_output = ""
    total_words = 0

    for section_name, items in notes.items():
        clean_section_name = section_name.replace("_", " ").replace("", "").strip()
        print(f"Synthesizing Section: {clean_section_name}...")
        
        raw_section_text = ""
        for item in items:
            raw_section_text += f"{item['content']} "
            
        synthesizer_prompt = SystemMessage(content="""You are an expert Science Olympiad Editor optimizing for EXTREME SPACE EFFICIENCY.
        Synthesize the provided notes into a single section.
        
        STRICT RULES:
        1. RETAIN ALL FACTS: Do not lose any formulas, numbers, or key terms.
        2. PURE CONTINUOUS PROSE: Write the entire synthesis as one single, massive, continuous block of text. ABSOLUTELY NO line breaks or new paragraphs.
        3. PERFECT GRAMMAR & PUNCTUATION: The raw notes contain fragmented bullet points. You MUST convert these fragments into clear, concise, and complete English sentences. Every distinct fact must end with a period or semicolon. Ensure the English is highly readable and not a confusing run-on.
        4. NO FORMATTING: Do not use bold text, italics, bullet points, or markdown. Use plain text ONLY.
        5. LATEX MANDATE: You MUST preserve all LaTeX formatting (e.g., $E=mc^2$) for formulas and chemical equations. Do NOT strip backslashes or dollar signs.
        6. NO EMOJIS.
        7. NO FILLER: Remove all introductory or transition sentences.""")
        
        synthesis_req = HumanMessage(content=f"Synthesize these disjointed notes into a cohesive block of plain text while PRESERVING LaTeX math:\n\n{raw_section_text}")
        
        try:
            # Ask AI to smooth it out
            synthesized_content = llm.invoke([synthesizer_prompt, synthesis_req]).content
            
            # Programmatic failsafe: forcefully strip any stray newlines or bold markers the AI might sneak in
            # CRITICAL: We NO LONGER strip single asterisks as they might be used in LaTeX math.
            synthesized_content = synthesized_content.replace("\n", " ").replace("**", "").strip()
            
            markdown_output += f"{synthesized_content} "
            total_words += len(synthesized_content.split())
            print("   ✅ Density maximized and section compiled.")
            
        except Exception as e:
            print(f"   ❌ Error synthesizing section: {e}")
            for item in items:
                text = item['content'].replace("\n", " ").replace("**", "").replace("*", "")
                markdown_output += f"{text} "
            markdown_output += "\n"

    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(markdown_output.strip())
        print(f"\n✅ SUCCESS! Final synthesized cheat sheet saved to '{OUTPUT_FILE}'!")
        print(f"Approximate Word Count: {total_words} words.")
    except Exception as e:
         print(f"❌ Error saving final file: {e}")
         
    return markdown_output.strip()

if __name__ == "__main__":
    run({})
