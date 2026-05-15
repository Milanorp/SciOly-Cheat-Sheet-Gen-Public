import os
import sys
import pymupdf4llm

# Add the project root to sys.path to allow imports if needed later
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 1. Point the tool at your specific PDF
pdf_path = r"sample_rules.pdf"

# 2. Convert the entire document to Markdown
print(f"Extracting {pdf_path}... this might take a few seconds.")
md_text = pymupdf4llm.to_markdown(pdf_path)

# 3. Save the output to a text file so you can inspect it
output_file = "extracted_rules.md"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(md_text)

print(f"Success! Open {output_file} to see exactly what the AI will read.")