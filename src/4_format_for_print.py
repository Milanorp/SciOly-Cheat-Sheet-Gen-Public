import os
from markdown_it import MarkdownIt
import subprocess

print("\n" + "="*60)
print("📄 PHASE 4: CHEAT SHEET FORMATTER (EXTREME DENSITY) 📄")
print("="*60)

INPUT_FILE = "Final_Cheat_Sheet.md"
OUTPUT_FILE = "Final_Cheat_Sheet_Printable.html"

try:
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        md_text = f.read()
except FileNotFoundError:
    print(f"❌ Error: '{INPUT_FILE}' not found. Please run '3_cheat_sheet_compiler.py' first.")
    exit(1)

# Convert Markdown to HTML
md = MarkdownIt("commonmark")
html_body = md.render(md_text)

# --- THE MAGIC CSS (Ultra-Density Edition) ---
html_template = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>SciOly Cheat Sheet</title>
<style>
    /* Reset margins for absolute maximum space */
    @page {{
        size: letter portrait;
        margin: 0.15in; /* Push text to the absolute physical printable edge */
    }}
    
    body {{
        font-family: "Segoe UI", Arial, sans-serif;
        font-size: 5.5pt; /* Extremely tiny, dense font to hit 3.5k+ words */
        line-height: 1.05; /* Squish lines together */
        margin: 0;
        padding: 0;
        text-align: justify; /* Make it look like a solid brick of text */
    }}

    /* Remove unnecessary titles and dividers */
    h1, hr {{
        display: none;
    }}
    
    /* Make section headers tiny and inline so they don't break the flow */
    h2 {{
        font-size: 6pt;
        display: inline;
        background-color: #ddd;
        color: black;
        border: 1px solid black;
        padding: 0px 2px;
        margin: 0 4px 0 0;
        text-transform: uppercase;
        font-weight: normal;
    }}

    /* Force EVERYTHING to stay on the same continuous line */
    p, ul, li, div, h3 {{
        display: inline;
        margin: 0;
        padding: 0;
    }}
    
    /* Strip all bold or italic styling just in case */
    strong, b, em, i {{
        font-weight: normal;
        font-style: normal;
    }}
</style>
</head>
<body>
    {html_body}
</body>
</html>
"""

# Save the HTML file
try:
    with open(OUTPUT_FILE, "w", encoding="utf-8", errors="xmlcharrefreplace") as f:
        f.write(html_template)
    print(f"✅ SUCCESS! Created ultra-dense layout: '{OUTPUT_FILE}'")
    
    # --- AUTOMATIC PDF CONVERSION ---
    print("🚀 Automatically converting HTML to PDF using browser headless engine...")
    pdf_output = "Final_Cheat_Sheet.pdf"
    edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    
    if os.path.exists(edge_path):
        command = [
            edge_path,
            "--headless",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={os.path.abspath(pdf_output)}",
            f"file:///{os.path.abspath(OUTPUT_FILE)}"
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"🎉 SUCCESS! Final ready-to-print PDF saved to: '{pdf_output}'")
    else:
        print("⚠️ Microsoft Edge not found. Could not auto-generate PDF. Please open the HTML file and print manually.")

except Exception as e:
    print(f"❌ Error creating printable file: {e}")
