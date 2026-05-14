import sys
import os
import asyncio

# Ensure src/ is in PYTHONPATH so internal imports work seamlessly
sys.path.append(os.path.abspath("src"))

from src import test_cruncher
from src import cheat_sheet_architect
from src import setup_cache
from src import research_dispatcher
from src import research_auditor
from src import cheat_sheet_compiler

def main():
    print("\n" + "="*70)
    print("SCIENCE OLYMPIAD AUTONOMOUS CHEAT SHEET GENERATOR")
    print("="*70)
    print("Initializing the 6-Phase Pipeline...")

    try:
        # Phase 0: Test Cruncher
        frequency_data = test_cruncher.run()

        # Get Event Name
        event_name = input("\nWhat Science Olympiad event are you building a cheat sheet for? ")

        # Phase 1: Architect
        event_name, blueprint = cheat_sheet_architect.run(event_name, frequency_data)

        if not blueprint:
            print("\n❌ Pipeline stopped: Architect could not generate a blueprint.")
            sys.exit(1)

        # Phase 1.5: Setup Cache
        cache_info = setup_cache.run()

        # Phase 2 & 2.5: Research and Self-Correction Loop
        if os.name == 'nt':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        # Start the loop
        max_retries = 3
        retry_count = 0
        failed_topics = None

        while retry_count < max_retries:
            research_notes = asyncio.run(research_dispatcher.run(event_name, blueprint, cache_info, target_topics=failed_topics))

            if not research_notes:
                print("\n❌ Pipeline stopped: No research notes were generated.")
                sys.exit(1)

            # Phase 2.5: Audit the research
            failed_topics = research_auditor.run(research_notes)

            if not failed_topics:
                break

            retry_count += 1
            print(f"\n🔄 Self-Correction Loop: Retry {retry_count}/{max_retries} for {len(failed_topics)} topics...")

        if failed_topics:
            print(f"\n⚠️ Warning: {len(failed_topics)} topics failed audit after {max_retries} retries. Proceeding with best available notes.")

        # Phase 3: Compiler
        latex_output = cheat_sheet_compiler.run(research_notes)
        
        if not latex_output:
            print("\n❌ Pipeline stopped: Compiler failed to generate LaTeX.")
            sys.exit(1)

        # Phase 4: Formatter
        format_for_print.run(latex_output)

        print("\n" + "="*70)
        print("PIPELINE COMPLETE!")
        print("Check 'Final_Cheat_Sheet.pdf' for your max-density competition sheet!")
        print("="*70)

    except KeyboardInterrupt:
        print("\n\n⚠️ Pipeline interrupted by user. Progress has been saved in checkpoint files!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Pipeline failed unexpectedly: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
