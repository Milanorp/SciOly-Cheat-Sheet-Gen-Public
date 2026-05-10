import subprocess
import sys
import os

print("\n" + "="*70)
print("🚀 SCIENCE OLYMPIAD AUTONOMOUS CHEAT SHEET GENERATOR 🚀")
print("="*70)
print("Initializing the 5-Phase Pipeline...")

# Ensure we are running from the root directory but calling scripts in src/
# We add src/ to PYTHONPATH so the scripts can easily import token_tracker
env = os.environ.copy()
env["PYTHONPATH"] = os.path.abspath("src") + os.pathsep + env.get("PYTHONPATH", "")

scripts = [
    ("Phase 1: The Architect", "src/1_cheat_sheet_architect.py"),
    ("Phase 1.5: Upload Rulebook Cache", "src/1.5_setup_cache.py"),
    ("Phase 2: The Research Dispatcher", "src/2_research_dispatcher.py"),
    ("Phase 3: The AI Compiler", "src/3_cheat_sheet_compiler.py"),
    ("Phase 4: The PDF Formatter", "src/4_format_for_print.py")
]

for phase_name, script_path in scripts:
    print(f"\n[{phase_name}]")
    print("-" * 50)
    
    # Run the script and stream its output to the terminal in real-time
    try:
        process = subprocess.Popen(
            [sys.executable, script_path],
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True
        )
        process.wait()
        
        if process.returncode != 0:
            print(f"\n❌ Error: {phase_name} stopped unexpectedly (Exit Code: {process.returncode}).")
            print("You can run 'python main.py' again later to resume progress!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Failed to execute {script_path}: {e}")
        sys.exit(1)

print("\n" + "="*70)
print("🎉 PIPELINE COMPLETE! 🎉")
print("Check 'Final_Cheat_Sheet.pdf' for your max-density competition sheet!")
print("="*70)
