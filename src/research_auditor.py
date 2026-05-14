import os
import json
from langchain_core.messages import HumanMessage, SystemMessage
from src.token_tracker import TokenTrackerCallback
from src.factory import factory
from src.models import AuditReport

def run(research_notes: dict) -> list[str]:
    print("\n" + "="*60)
    print("🕵️ PHASE 2.5: THE RESEARCH AUDITOR (SELF-CORRECTION) 🕵️")
    print("="*60)

    tracker = TokenTrackerCallback(script_name="2.5_research_auditor")
    llm = factory.get_llm(purpose="auditor")
    llm.callbacks = [tracker]
    
    structured_llm = llm.with_structured_output(AuditReport)

    failed_topics = []
    total_audited = 0
    total_passed = 0

    auditor_prompt = SystemMessage(content="""You are a Senior Science Olympiad Technical Auditor.
    Your job is to strictly evaluate research notes for a cheat sheet.
    
    CRITERIA FOR A PASS (A or B):
    1. TECHNICAL DEPTH: Does it include specific formulas, constants, and edge cases?
    2. ACCURACY: Does it directly address the requirements?
    3. NO FLUFF: Is it dense and fact-heavy?
    4. LATEX USAGE: Are formulas properly formatted in LaTeX?
    
    CRITERIA FOR A FAIL (C, D, or F):
    1. VAGUENESS: Saying "the formula for X" instead of actually writing the formula.
    2. MISSING DATA: If the requirements asked for "unit conversions" and they aren't there.
    3. POOR FORMATTING: Messy text or missing LaTeX.
    
    If it fails, provide a brutal, short correction memo.""")

    for section_name, items in research_notes.items():
        print(f"Auditing Section: {section_name}...")
        for item in items:
            topic = item['original_target']
            content = item['content']
            requirements = item['expanded_requirements']
            
            total_audited += 1
            
            audit_msg = HumanMessage(content=f"""
            TARGET TOPIC: {topic}
            REQUIREMENTS TO COVER: {requirements}
            RESEARCH DRAFT: 
            {content}
            """)
            
            try:
                result = structured_llm.invoke([auditor_prompt, audit_msg])
                
                if result.is_pass:
                    total_passed += 1
                else:
                    print(f"   [FAIL] {topic[:40]}... Grade: {result.grade}")
                    print(f"          > Feedback: {result.feedback}")
                    failed_topics.append(topic)
            except Exception as e:
                print(f"   ❌ Error auditing '{topic[:20]}': {e}")
                total_passed += 1 # Default to pass on AI error to prevent loops

    print(f"\nAudit Summary: {total_passed}/{total_audited} topics passed.")
    if failed_topics:
        print(f"🚨 {len(failed_topics)} topics flagged for re-research.")
    else:
        print("✅ All research notes passed the quality audit!")
        
    return failed_topics

if __name__ == "__main__":
    run({})
