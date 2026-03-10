import os
import re
import json
from pathlib import Path
import pdfplumber
from markdownify import markdownify as md


# ------------------------------------------------
# PATH CONFIGURATION
# ------------------------------------------------

BASE_DIR = Path("/Users/pranjalsharma/Desktop/secfilling")

INPUT_FOLDER = BASE_DIR / "pdf_documents"
OUTPUT_FOLDER = BASE_DIR / "processed_documents"

OUTPUT_FOLDER.mkdir(exist_ok=True)


# ------------------------------------------------
# DOCUMENT TYPE DETECTION
# ------------------------------------------------

def detect_document_type(filename, text):

    text = text.lower()
    filename = filename.lower()

    if "form 10-k" in text or "annual report" in filename:
        return "10-K"

    if "form 10-q" in text or "q1" in filename or "q2" in filename or "q3" in filename:
        return "10-Q"

    return "UNKNOWN"


# ------------------------------------------------
# TICKER EXTRACTION
# ------------------------------------------------

def extract_ticker(text):

    pattern = r"trading symbol[s]?\s*[:\-]?\s*([A-Z]{1,5})"
    match = re.search(pattern, text, re.IGNORECASE)

    if match:
        return match.group(1)

    return "UNKNOWN"


# ------------------------------------------------
# FISCAL YEAR EXTRACTION
# ------------------------------------------------

def extract_fiscal_year(text):

    match = re.search(r"fiscal year ended\s+.*?(\d{4})", text, re.IGNORECASE)

    if match:
        return match.group(1)

    return "UNKNOWN"


# ------------------------------------------------
# QUARTER DETECTION
# ------------------------------------------------

def extract_quarter(filename):

    filename = filename.lower()

    if "q1" in filename:
        return "Q1"

    if "q2" in filename:
        return "Q2"

    if "q3" in filename:
        return "Q3"

    if "q4" in filename:
        return "Q4"

    return None


# ------------------------------------------------
# SAFE TABLE -> MARKDOWN
# ------------------------------------------------

def convert_table_to_markdown(table):

    if not table or len(table) == 0:
        return ""

    headers = [str(h) if h else "" for h in table[0]]

    md_table = []

    md_table.append("| " + " | ".join(headers) + " |")
    md_table.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for row in table[1:]:

        clean_row = [str(cell) if cell else "" for cell in row]

        md_table.append("| " + " | ".join(clean_row) + " |")

    return "\n".join(md_table)


# ------------------------------------------------
# PDF PROCESSOR
# ------------------------------------------------

def process_pdf(pdf_path):

    document_id = Path(pdf_path).stem
    pages_output = []

    with pdfplumber.open(pdf_path) as pdf:

        full_text_sample = ""

        for i, page in enumerate(pdf.pages):

            page_text = page.extract_text() or ""

            full_text_sample += page_text[:2000]

            tables = page.extract_tables()

            markdown_tables = []

            if tables:

                for table in tables:

                    try:
                        md_table = convert_table_to_markdown(table)

                        if md_table:
                            markdown_tables.append(md_table)

                    except Exception:
                        continue

            page_md = md(page_text)

            if markdown_tables:
                page_md += "\n\n" + "\n\n".join(markdown_tables)

            page_object = {
                "Document_ID": document_id,
                "Page_Index": i + 1,
                "Content_MD": page_md
            }

            pages_output.append(page_object)

    # ------------------------------------------------
    # METADATA
    # ------------------------------------------------

    doc_type = detect_document_type(document_id, full_text_sample)
    ticker = extract_ticker(full_text_sample)
    fiscal_year = extract_fiscal_year(full_text_sample)
    quarter = extract_quarter(document_id)

    for page in pages_output:

        page["Ticker"] = ticker
        page["Fiscal_Year"] = fiscal_year
        page["Document_Type"] = doc_type
        page["Quarter"] = quarter

        page["Tags"] = [
            "SEC_Filing",
            doc_type,
            ticker,
            fiscal_year
        ]

    return pages_output


# ------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------

def run_pipeline():

    all_documents = []

    for pdf_file in os.listdir(INPUT_FOLDER):

        if pdf_file.endswith(".pdf"):

            pdf_path = INPUT_FOLDER / pdf_file

            print(f"Processing: {pdf_file}")

            processed_pages = process_pdf(pdf_path)

            all_documents.extend(processed_pages)

    output_file = OUTPUT_FOLDER / "processed_sec_filings.json"

    with open(output_file, "w") as f:
        json.dump(all_documents, f, indent=2)

    print("\nProcessing Complete")
    print(f"Saved at: {output_file}")


# ------------------------------------------------
# ENTRY POINT
# ------------------------------------------------

if __name__ == "__main__":
    run_pipeline()
