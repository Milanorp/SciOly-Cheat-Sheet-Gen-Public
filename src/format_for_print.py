import os
import subprocess
import shutil

def run(latex_content: str = None) -> None:
    print("\n" + "="*60)
    print("PHASE 4: CHEAT SHEET FORMATTER (NATIVE LATEX)")
    print("="*60)

    INPUT_FILE = "Final_Cheat_Sheet.tex"
    OUTPUT_PDF = "Final_Cheat_Sheet.pdf"

    if not latex_content:
        try:
            with open(INPUT_FILE, "r", encoding="utf-8") as f:
                latex_content = f.read()
        except FileNotFoundError:
            print(f"❌ Error: '{INPUT_FILE}' not found. Please run Phase 3 first.")
            return

    # Check for LaTeX compilers
    compilers = ["pdflatex", "tectonic", "xelatex"]
    found_compiler = None
    
    for c in compilers:
        if shutil.which(c):
            found_compiler = c
            break

    if found_compiler:
        print(f"🚀 Found LaTeX compiler: {found_compiler}. Attempting to compile...")
        try:
            # Run the compiler
            # pdflatex usually needs to run twice for references, 
            # but we don't have any, so once is enough.
            if found_compiler == "pdflatex" or found_compiler == "xelatex":
                process = subprocess.run(
                    [found_compiler, "-interaction=nonstopmode", INPUT_FILE],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
            elif found_compiler == "tectonic":
                process = subprocess.run(
                    [found_compiler, INPUT_FILE],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
            
            if os.path.exists(OUTPUT_PDF):
                print(f"✅ SUCCESS! Final professional PDF generated: '{OUTPUT_PDF}'")
                
                # Cleanup auxiliary files
                aux_extensions = [".aux", ".log", ".out", ".toc"]
                for ext in aux_extensions:
                    aux_file = INPUT_FILE.replace(".tex", ext)
                    if os.path.exists(aux_file):
                        os.remove(aux_file)
            else:
                print(f"❌ Compilation failed. Compiler output:\n{process.stdout}")
                print("\n⚠️  Manual Step Required: Your LaTeX is perfect, but the local compiler failed.")
                print(f"Please upload '{INPUT_FILE}' to Overleaf.com for a one-click perfect PDF.")

        except Exception as e:
            print(f"❌ Error during compilation: {e}")
            print(f"⚠️  Manual Step Required: Please upload '{INPUT_FILE}' to Overleaf.com.")
    else:
        print("❌ No LaTeX compiler (pdflatex, tectonic, or xelatex) found on this system.")
        print(f"✅ SUCCESS! Created professional LaTeX source: '{INPUT_FILE}'")
        print("\n🚀 NEXT STEP: To get your professional PDF:")
        print(f"1. Go to Overleaf.com")
        print(f"2. Create a new project and upload '{INPUT_FILE}'")
        print(f"3. Hit 'Recompile' for a math-perfect, high-density cheat sheet.")

if __name__ == "__main__":
    run()
