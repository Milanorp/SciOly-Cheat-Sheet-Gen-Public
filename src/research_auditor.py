import os
import json
from langchain_core.messages import HumanMessage, SystemMessage
from src.token_tracker import TokenTrackerCallback
from src.factory import factory, console
from src.models import AuditReport
from rich.panel import Panel

def run(research_notes: dict) -> list[str]:
    console.print("\n[phase]🕵️ PHASE 2.5: THE RESEARCH AUDITOR (SELF-CORRECTION) 🕵️[/phase]")

    tracker = TokenTrackerCallback(script_name="2.5_research_auditor")
    llm = factory.get_llm(purpose="auditor")
    llm.callbacks = [tracker]
    
    structured_llm = llm.with_structured_output(AuditReport)

    failed_topics = []
    total_audited = 0
    total_passed = 0

    auditor_prompt = SystemMessage(content="""You are a Senior Science Olympiad Technical Auditor.
    Evaluate research notes for density, accuracy, LaTeX usage, and completeness.""")

    for section_name, items in research_notes.items():
        console.print(f"\n[cyan]Auditing Section:[/cyan] {section_name}")
        for item in items:
            topic = item['original_target']
            content = item['content']
            requirements = item['expanded_requirements']
            total_audited += 1
            
            audit_msg = HumanMessage(content=f"TARGET: {topic}\nREQUIREMENTS: {requirements}\nDRAFT:\n{content}")
            
            try:
                result = structured_llm.invoke([auditor_prompt, audit_msg])
                
                if result.is_pass:
                    total_passed += 1
                    console.print(f"   [[success]PASS[/success]] {topic[:40]}...")
                else:
                    console.print(Panel(
                        f"[warning]Feedback:[/warning] {result.feedback}",
                        title=f"[error]FAIL[/error] {topic[:40]}...",
                        border_style="red"
                    ))
                    failed_topics.append(topic)
            except Exception as e:
                console.print(f"[error]   ❌ Error auditing '{topic[:20]}': {e}[/error]")
                total_passed += 1

    console.print(f"\n[info]Audit Summary: {total_passed}/{total_audited} topics passed.[/info]")
    if failed_topics:
        console.print(f"[warning]🚨 {len(failed_topics)} topics flagged for re-research.[/warning]")
    else:
        console.print("[success]✅ All research notes passed the quality audit![/success]")
        
    return failed_topics

if __name__ == "__main__":
    run({})
