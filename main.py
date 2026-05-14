import sys
import os
import asyncio
from src.state_manager import state_manager
from src import test_cruncher, cheat_sheet_architect, setup_cache, research_dispatcher, research_auditor, cheat_sheet_compiler, format_for_print
from src.factory import factory, console
from rich.prompt import Prompt
from rich.panel import Panel

# Ensure src/ is in PYTHONPATH
sys.path.append(os.path.abspath("src"))

async def run_pipeline():
    console.print(Panel.fit(
        "[bold cyan]SCIENCE OLYMPIAD AUTONOMOUS CHEAT SHEET GENERATOR v2.5[/bold cyan]\n"
        "[dim]Professional Architecture & LaTeX Typesetting Engine[/dim]",
        border_style="cyan"
    ))
    
    state = state_manager.load_state()
    config = factory.get_config()
    
    try:
        # --- PHASE 0: TEST CRUNCHER ---
        if state.current_phase <= 0:
            state.frequency_data = test_cruncher.run()
            state.current_phase = 1.0
            state_manager.save_state(state)

        # --- EVENT NAME INPUT ---
        if not state.event_name:
            state.event_name = Prompt.ask("\n[bold cyan]What Science Olympiad event are you building a cheat sheet for?[/bold cyan]")
            state_manager.save_state(state)

        # --- PHASE 1: ARCHITECT ---
        if state.current_phase <= 1:
            state.event_name, state.blueprint = cheat_sheet_architect.run(state.event_name, state.frequency_data)
            if not state.blueprint:
                console.print("\n[error]❌ Pipeline stopped: Architect failed.[/error]")
                return
            state.current_phase = 1.5
            state_manager.save_state(state)

        # --- PHASE 1.5: SETUP CACHE ---
        if state.current_phase <= 1.5:
            state.cache_info = setup_cache.run()
            state.current_phase = 2.0
            state_manager.save_state(state)

        # --- PHASE 2 & 2.5: RESEARCH & AUDIT LOOP ---
        if state.current_phase <= 2:
            if os.name == 'nt':
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            
            max_retries = config['research']['max_audit_retries']
            
            while state.retry_count < max_retries:
                targets = state.failed_topics if state.failed_topics else None
                state.research_notes = await research_dispatcher.run(
                    state.event_name, 
                    state.blueprint, 
                    state.cache_info, 
                    target_topics=targets
                )
                
                if not state.research_notes:
                    console.print("\n[error]❌ Pipeline stopped: Research failed.[/error]")
                    return
                
                state.failed_topics = research_auditor.run(state.research_notes)
                
                if not state.failed_topics:
                    break
                
                state.retry_count += 1
                state_manager.save_state(state)
                console.print(f"\n[warning]🔄 Self-Correction Loop: Retry {state.retry_count}/{max_retries} for {len(state.failed_topics)} topics...[/warning]")

            state.current_phase = 3.0
            state_manager.save_state(state)

        # --- PHASE 3: COMPILER ---
        if state.current_phase <= 3:
            latex_output = cheat_sheet_compiler.run(state.research_notes)
            if not latex_output:
                console.print("\n[error]❌ Pipeline stopped: Compiler failed.[/error]")
                return
            state.current_phase = 4.0
            state_manager.save_state(state)

        # --- PHASE 4: FORMATTER ---
        if state.current_phase <= 4:
            format_for_print.run()
            state.current_phase = 5.0 # Done
            state_manager.save_state(state)

        console.print(Panel(
            f"[success]Check '{config['paths']['output_pdf']}' for your max-density competition sheet![/success]",
            title="[success]PIPELINE COMPLETE![/success]",
            border_style="green"
        ))

    except KeyboardInterrupt:
        console.print("\n\n[warning]⚠️ Pipeline interrupted by user. Progress saved.[/warning]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[error]❌ Pipeline failed unexpectedly: {e}[/error]")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(run_pipeline())
