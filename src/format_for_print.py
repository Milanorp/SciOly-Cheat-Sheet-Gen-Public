import os
try:
    import typst
except ImportError:
    typst = None
from src.factory import factory

def run(typst_content: str = None) -> None:
    config = factory.get_config()
    INPUT_FILE = config['paths']['output_typ']
    OUTPUT_PDF = config['paths']['output_pdf']

    print("\n" + "="*60)
    print("PHASE 4: CHEAT SHEET FORMATTER (TYPST)")
    print("="*60)

    if not typst_content:
        try:
            with open(INPUT_FILE, "r", encoding="utf-8") as f:
                typst_content = f.read()
        except FileNotFoundError:
            print(f"❌ Error: '{INPUT_FILE}' not found. Please run Phase 3 first.")
            return

    if typst is None:
        print("❌ Error: 'typst' python package is not installed.")
        print("Please run 'pip install typst' to compile PDFs locally.")
        return

    print("🚀 Found Typst Python Compiler. Attempting to compile...")
    try:
        # Compile the Typst file into a PDF natively in Python!
        typst.compile(INPUT_FILE, output=OUTPUT_PDF)
        
        if os.path.exists(OUTPUT_PDF):
            print(f"✅ SUCCESS! Final professional PDF generated: '{OUTPUT_PDF}'")
        else:
            print("❌ Compilation failed silently. PDF not found.")
            
    except Exception as e:
        print(f"❌ Error during Typst compilation: {e}")
        print(f"⚠️  Please review '{INPUT_FILE}' for syntax errors.")

if __name__ == "__main__":
    run()
