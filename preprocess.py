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

    if "form 10-k" in text or "annual report" in filename or "10-k" in filename:
        return "10-K"

    if "form 10-q" in text or "10-q" in filename or any(
        q in filename for q in ["q1", "q2", "q3"]
    ):
        return "10-Q"

    if "form 8-k" in text or "8-k" in filename:
        return "8-K"

    if "proxy" in text or "def 14a" in filename:
        return "DEF14A"

    return "UNKNOWN"


# ------------------------------------------------
# TICKER EXTRACTION
# ------------------------------------------------

def extract_ticker(text):

    # Primary: trading symbol line
    pattern = r"trading symbol[s]?\s*[:\-]?\s*([A-Z]{1,5})"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Secondary: "NASDAQ: TICK" or "NYSE: TICK"
    pattern2 = r"(?:NASDAQ|NYSE|NYSEARCA)[:\s]+([A-Z]{1,5})"
    match2 = re.search(pattern2, text, re.IGNORECASE)
    if match2:
        return match2.group(1).upper()

    # Tertiary: common stock ticker pattern on cover page
    pattern3 = r"Common Stock.*?\(([A-Z]{2,5})\)"
    match3 = re.search(pattern3, text[:3000])
    if match3:
        return match3.group(1).upper()

    return "UNKNOWN"


# ------------------------------------------------
# FISCAL YEAR EXTRACTION
# ------------------------------------------------

def extract_fiscal_year(text):

    # "fiscal year ended Month DD, YYYY"
    match = re.search(
        r"fiscal year ended\s+(?:\w+\s+\d{1,2},?\s+)?(\d{4})",
        text, re.IGNORECASE
    )
    if match:
        return match.group(1)

    # "for the year ended December 31, 2024"
    match2 = re.search(
        r"for the year ended\s+\w+\s+\d{1,2},?\s+(\d{4})",
        text, re.IGNORECASE
    )
    if match2:
        return match2.group(1)

    # Last 4-digit year that looks like a filing year
    years = re.findall(r"\b(20\d{2})\b", text[:5000])
    if years:
        return max(set(years), key=years.count)

    return "UNKNOWN"


# ------------------------------------------------
# QUARTER DETECTION
# ------------------------------------------------

def extract_quarter(filename, text=""):

    filename = filename.lower()

    for q in ["q1", "q2", "q3", "q4"]:
        if q in filename:
            return q.upper()

    # Try from text: "three months ended March 31"
    quarter_map = {
        r"three months ended\s+(?:march|january|february)\s+\d": "Q1",
        r"three months ended\s+(?:june|april|may)\s+\d": "Q2",
        r"three months ended\s+(?:september|july|august)\s+\d": "Q3",
        r"three months ended\s+(?:december|october|november)\s+\d": "Q4",
    }
    for pattern, quarter in quarter_map.items():
        if re.search(pattern, text.lower()):
            return quarter

    return None


# ------------------------------------------------
# FINANCIAL SECTION DETECTION
# ------------------------------------------------

FINANCIAL_SECTION_PATTERNS = {
    "revenue":           r"\brevenue\b|\bnet revenue\b|\btotal revenue\b",
    "income_statement":  r"\bincome from operations\b|\boperating income\b|\bnet income\b|\bnet loss\b",
    "balance_sheet":     r"\btotal assets\b|\btotal liabilities\b|\bstockholders.{0,10}equity\b",
    "cash_flow":         r"\bcash flow\b|\boperating activities\b|\bfree cash flow\b",
    "guidance":          r"\bguidance\b|\boutlook\b|\bfiscal year \d{4}\b",
    "margin":            r"\bgross margin\b|\boperating margin\b|\bnet margin\b|\bebitda\b",
    "segment":           r"\bsegment\b|\bgeograph\b|\bregion\b",
    "risk":              r"\brisk factor\b|\bmaterial weakness\b|\blitigation\b",
    "mda":               r"\bmanagement.{0,15}discussion\b|\bmd&a\b",
    "earnings_per_share":r"\bearnings per share\b|\beps\b|\bdiluted\b",
}

def detect_financial_tags(text):
    """Return list of financial topic tags found in page text."""
    tags = []
    text_lower = text.lower()
    for tag, pattern in FINANCIAL_SECTION_PATTERNS.items():
        if re.search(pattern, text_lower):
            tags.append(tag)
    return tags


# ------------------------------------------------
# SAFE TABLE -> MARKDOWN
# ------------------------------------------------

def convert_table_to_markdown(table):

    if not table or len(table) == 0:
        return ""

    # Clean None values
    cleaned = []
    for row in table:
        cleaned.append([str(cell).strip() if cell is not None else "" for cell in row])

    if not cleaned:
        return ""

    headers = cleaned[0]
    md_table = []
    md_table.append("| " + " | ".join(headers) + " |")
    md_table.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for row in cleaned[1:]:
        # Pad short rows
        while len(row) < len(headers):
            row.append("")
        md_table.append("| " + " | ".join(row[:len(headers)]) + " |")

    return "\n".join(md_table)


# ------------------------------------------------
# EXTRACT FINANCIAL NUMBERS FROM PAGE
# ------------------------------------------------

def extract_key_numbers(text):
    """Pull dollar amounts and percentages for quick lookup."""
    numbers = {}

    # Dollar amounts like $1.2B, $450M, $1,234,567
    dollar_pattern = r"\$\s?(\d[\d,\.]*)\s?(billion|million|thousand|B|M|K)?"
    for match in re.finditer(dollar_pattern, text, re.IGNORECASE):
        amount_str = match.group(0)
        # Store up to 5 unique dollar figures
        if len(numbers.get("dollar_amounts", [])) < 5:
            numbers.setdefault("dollar_amounts", []).append(amount_str.strip())

    # Percentages
    pct_pattern = r"(\d+\.?\d*)\s?%"
    pcts = re.findall(pct_pattern, text)
    if pcts:
        numbers["percentages"] = pcts[:5]

    # Year-over-year growth signals
    yoy_pattern = r"(?:increased?|decreased?|grew?|declined?|rose|fell)\s+(?:by\s+)?(\d+\.?\d*)\s?%"
    yoy = re.findall(yoy_pattern, text, re.IGNORECASE)
    if yoy:
        numbers["yoy_changes_pct"] = yoy[:3]

    return numbers


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

            # Extract tables
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

            # Financial tag detection per page
            financial_tags = detect_financial_tags(page_text)
            key_numbers = extract_key_numbers(page_text)

            page_object = {
                "Document_ID":     document_id,
                "Page_Index":      i + 1,
                "Content_MD":      page_md,
                "Financial_Tags":  financial_tags,
                "Key_Numbers":     key_numbers,
                "Char_Count":      len(page_text),
            }

            pages_output.append(page_object)

    # ------------------------------------------------
    # METADATA — computed once from full sample
    # ------------------------------------------------

    doc_type    = detect_document_type(document_id, full_text_sample)
    ticker      = extract_ticker(full_text_sample)
    fiscal_year = extract_fiscal_year(full_text_sample)
    quarter     = extract_quarter(document_id, full_text_sample)

    for page in pages_output:
        page["Ticker"]        = ticker
        page["Fiscal_Year"]   = fiscal_year
        page["Document_Type"] = doc_type
        page["Quarter"]       = quarter
        page["Tags"]          = list(set([
            "SEC_Filing", doc_type, ticker, fiscal_year
        ] + page["Financial_Tags"]))

    print(f"  → {len(pages_output)} pages | Ticker: {ticker} | "
          f"Type: {doc_type} | Year: {fiscal_year}")

    return pages_output


# ------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------

def run_pipeline():

    pdf_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"No PDFs found in {INPUT_FOLDER}")
        return

    all_documents = []

    for pdf_file in pdf_files:
        pdf_path = INPUT_FOLDER / pdf_file
        print(f"\nProcessing: {pdf_file}")
        try:
            processed_pages = process_pdf(pdf_path)
            all_documents.extend(processed_pages)
        except Exception as e:
            print(f"  ERROR processing {pdf_file}: {e}")

    output_file = OUTPUT_FOLDER / "processed_sec_filings.json"

    with open(output_file, "w") as f:
        json.dump(all_documents, f, indent=2)

    print(f"\nProcessing Complete — {len(all_documents)} pages total")
    print(f"Saved at: {output_file}")


# ------------------------------------------------
# ENTRY POINT
# ------------------------------------------------

if __name__ == "__main__":
    run_pipeline()
