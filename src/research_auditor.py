import os
import json
import asyncio
import aiofiles
from langchain_core.messages import HumanMessage, SystemMessage
from src.token_tracker import TokenTrackerCallback
from src.factory import factory, console
from src.models import AuditReport, ResearchNote
from rich.panel import Panel

def run(research_notes: dict) -> list[str]:
    config = factory.get_config()
    DATA_DIR = config['paths']['data_dir']
    NOTES_FILE = os.path.join(DATA_DIR, "raw_research_notes.json")

    console.print("\n[phase]🕵️ PHASE 2.5: THE RESEARCH AUDITOR (SELF-CORRECTION) 🕵️[/phase]")

    tracker = TokenTrackerCallback(script_name="2.5_research_auditor")
    llm = factory.get_llm(purpose="auditor")
    llm.callbacks = [tracker]
    
    structured_llm = llm.with_structured_output(AuditReport)

    failed_topics = []
    total_to_audit = 0
    total_passed = 0
    skipped_count = 0

    auditor_prompt = SystemMessage(content="""You are a Senior Science Olympiad Technical Auditor.
    Evaluate research notes for density, accuracy, LaTeX usage, and completeness.
    STRICT AUDIT CRITERIA:
    1. CHEMISTRY: All chemical formulas MUST use \ce{...}. Fail if they use $\text{...}$ or plain text.
    2. MATH: Prefer inline math $...$. Fail if space-wasting block math $$...$$ is used.
    3. DENSITY: Content must be fact-dense with zero filler words.
    4. ACCURACY: Formulas and constants must be verified.""")

    for section_name, items in research_notes.items():
        console.print(f"\n[cyan]Auditing Section:[/cyan] {section_name}")
        for item in items:
            # Handle both raw dict and Pydantic-style dict
            is_verified = item.get('is_verified', False)
            topic = item['original_target']
            
            if is_verified:
                console.print(f"   [[success]VERIFIED[/success]] {topic[:40]}... (Skipping)")
                skipped_count += 1
                total_passed += 1
                continue

            content = item['content']
            requirements = item['expanded_requirements']
            total_to_audit += 1
            
            audit_msg = HumanMessage(content=f"TARGET: {topic}\nREQUIREMENTS: {requirements}\nDRAFT:\n{content}")
            
            try:
                result = structured_llm.invoke([auditor_prompt, audit_msg])
                
                if result.is_pass:
                    total_passed += 1
                    item['is_verified'] = True
                    console.print(f"   [[success]PASS[/success]] {topic[:40]}...")
                else:
                    console.print(Panel(
                        f"[warning]Feedback:[/warning] {result.feedback}",
                        title=f"[error]FAIL[/error] {topic[:40]}...",
                        border_style="red"
                    ))
                    item['is_verified'] = False
                    failed_topics.append(topic)
            except Exception as e:
                console.print(f"[error]   ❌ Error auditing '{topic[:20]}': {e}[/error]")
                total_passed += 1 # Prevent loops on AI error

    # Save the updated is_verified status back to the JSON file
    try:
        with open(NOTES_FILE, "w", encoding="utf-8") as f:
            json.dump(research_notes, f, indent=4)
    except Exception as e:
        console.print(f"[error]❌ Failed to save verified status: {e}[/error]")

    console.print(f"\n[info]Audit Summary: {total_passed}/{len(research_notes)*len(research_notes[section_name]) if research_notes else 0} topics passed.[/info]")
    console.print(f"   (Audited: {total_to_audit}, Already Verified: {skipped_count}, Newly Failed: {len(failed_topics)})")
    
    if failed_topics:
        console.print(f"[warning]🚨 {len(failed_topics)} topics flagged for re-research.[/warning]")
    else:
        console.print("[success]✅ All research notes passed the quality audit![/success]")
        
    return failed_topics

if __name__ == "__main__":
    run({})
