"""
preprocess.py — LAD-RAG Ingestion Pipeline
Layout-Aware Parser: extracts text, tables with spatial (X,Y) coordinates,
detects sections/headers, builds Table Knowledge Graph per document.
Architecture: SEC PDFs → Layout-Aware Parser → Hybrid Storage JSON
"""

import os
import re
import json
from pathlib import Path
import pdfplumber
from markdownify import markdownify as md


# ------------------------------------------------
# PATH CONFIGURATION
# ------------------------------------------------

BASE_DIR      = Path("/Users/pranjalsharma/Desktop/secfilling")
INPUT_FOLDER  = BASE_DIR / "pdf_documents"
OUTPUT_FOLDER = BASE_DIR / "processed_documents"
OUTPUT_FOLDER.mkdir(exist_ok=True)


# ------------------------------------------------
# DOCUMENT TYPE DETECTION
# ------------------------------------------------

def detect_document_type(filename, text):
    text     = text.lower()
    filename = filename.lower()
    if "form 10-k" in text or "annual report" in filename or "10-k" in filename:
        return "10-K"
    if "form 10-q" in text or "10-q" in filename or any(q in filename for q in ["q1","q2","q3"]):
        return "10-Q"
    if "form 8-k" in text or "8-k" in filename:
        return "8-K"
    if "proxy" in text or "def 14a" in filename:
        return "DEF14A"
    return "UNKNOWN"


# ------------------------------------------------
# TICKER / YEAR / QUARTER EXTRACTION
# ------------------------------------------------

def extract_ticker(text):
    for pattern in [
        r"trading symbol[s]?\s*[:\-]?\s*([A-Z]{1,5})",
        r"(?:NASDAQ|NYSE|NYSEARCA)[:\s]+([A-Z]{1,5})",
        r"Common Stock.*?\(([A-Z]{2,5})\)",
    ]:
        m = re.search(pattern, text[:5000], re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return "UNKNOWN"

def extract_fiscal_year(text):
    for pattern in [
        r"fiscal year ended\s+(?:\w+\s+\d{1,2},?\s+)?(\d{4})",
        r"for the year ended\s+\w+\s+\d{1,2},?\s+(\d{4})",
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1)
    years = re.findall(r"\b(20\d{2})\b", text[:5000])
    return max(set(years), key=years.count) if years else "UNKNOWN"

def extract_quarter(filename, text=""):
    filename = filename.lower()
    for q in ["q1","q2","q3","q4"]:
        if q in filename:
            return q.upper()
    qmap = {
        r"three months ended\s+(?:march|january|february)\s+\d": "Q1",
        r"three months ended\s+(?:june|april|may)\s+\d":         "Q2",
        r"three months ended\s+(?:september|july|august)\s+\d":  "Q3",
        r"three months ended\s+(?:december|october|november)\s+\d": "Q4",
    }
    for pat, q in qmap.items():
        if re.search(pat, text.lower()):
            return q
    return None


# ------------------------------------------------
# FINANCIAL TAG DETECTION
# ------------------------------------------------

FINANCIAL_SECTION_PATTERNS = {
    "revenue":            r"\brevenue\b|\bnet revenue\b|\btotal revenue\b",
    "income_statement":   r"\bincome from operations\b|\boperating income\b|\bnet income\b|\bnet loss\b",
    "balance_sheet":      r"\btotal assets\b|\btotal liabilities\b|\bstockholders.{0,10}equity\b",
    "cash_flow":          r"\bcash flow\b|\boperating activities\b|\bfree cash flow\b",
    "guidance":           r"\bguidance\b|\boutlook\b|\bfiscal year \d{4}\b",
    "margin":             r"\bgross margin\b|\boperating margin\b|\bnet margin\b|\bebitda\b",
    "segment":            r"\bsegment\b|\bgeograph\b|\bregion\b",
    "risk":               r"\brisk factor\b|\bmaterial weakness\b|\blitigation\b",
    "mda":                r"\bmanagement.{0,15}discussion\b|\bmd&a\b",
    "earnings_per_share": r"\bearnings per share\b|\beps\b|\bdiluted\b",
    "capex":              r"\bcapital expenditure\b|\bcapex\b",
    "debt":               r"\blong.term debt\b|\bborrowings\b|\bcredit facility\b",
}

def detect_financial_tags(text):
    tl = text.lower()
    return [tag for tag, pat in FINANCIAL_SECTION_PATTERNS.items() if re.search(pat, tl)]

def extract_key_numbers(text):
    nums = {}
    for m in re.finditer(r"\$\s?(\d[\d,\.]*)\s?(billion|million|thousand|[BMK])?", text, re.IGNORECASE):
        nums.setdefault("dollar_amounts", [])
        if len(nums["dollar_amounts"]) < 8:
            nums["dollar_amounts"].append(m.group(0).strip())
    pcts = re.findall(r"[\+\-]?\d+\.?\d*\s?%", text)
    if pcts:
        nums["percentages"] = pcts[:8]
    yoy = re.findall(
        r"(?:increased?|decreased?|grew?|declined?|rose|fell)\s+(?:by\s+)?(\d+\.?\d*)\s?%",
        text, re.IGNORECASE
    )
    if yoy:
        nums["yoy_changes_pct"] = yoy[:5]
    return nums


# ------------------------------------------------
# LAYOUT-AWARE TABLE EXTRACTION WITH SPATIAL COORDS
# Implements: "Preserve (X,Y) Spatial Coordinates"
# ------------------------------------------------

def extract_tables_with_layout(page):
    """
    Extract tables with bounding box (x0,y0,x1,y1) spatial coordinates.
    Returns list of dicts: {markdown, bbox, table_index, headers, row_count}
    """
    results      = []
    raw_tables   = page.extract_tables()
    try:
        table_objects = page.find_tables()
    except Exception:
        table_objects = []

    for i, table in enumerate(raw_tables):
        if not table:
            continue

        # Spatial bounding box
        bbox = None
        if i < len(table_objects):
            try:
                b    = table_objects[i].bbox
                bbox = {"x0": round(b[0],1), "y0": round(b[1],1),
                        "x1": round(b[2],1), "y1": round(b[3],1)}
            except Exception:
                pass

        cleaned = []
        for row in table:
            cleaned.append([str(c).strip() if c is not None else "" for c in row])
        if not cleaned:
            continue

        headers = cleaned[0]
        lines   = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"]*len(headers)) + " |",
        ]
        for row in cleaned[1:]:
            while len(row) < len(headers):
                row.append("")
            lines.append("| " + " | ".join(row[:len(headers)]) + " |")

        results.append({
            "table_index": i,
            "markdown":    "\n".join(lines),
            "headers":     headers,
            "bbox":        bbox,
            "row_count":   len(cleaned) - 1,
        })

    return results


# ------------------------------------------------
# HEADER / PARAGRAPH DETECTION
# Implements: "Detect Sections, Headers, Paragraphs"
# ------------------------------------------------

def detect_headers_and_paragraphs(page):
    """Use font-size heuristics to identify headers vs body text."""
    elements = []
    try:
        words = page.extract_words(extra_attrs=["size"])
        if not words:
            return elements
        sizes       = [w.get("size", 10) for w in words if w.get("size")]
        if not sizes:
            return elements
        median_size = sorted(sizes)[len(sizes)//2]

        lines = {}
        for w in words:
            y_key = round(w["top"] / 5) * 5
            lines.setdefault(y_key, []).append(w)

        for y_key in sorted(lines.keys()):
            lw        = sorted(lines[y_key], key=lambda w: w["x0"])
            line_text = " ".join(w["text"] for w in lw)
            avg_size  = sum(w.get("size",10) for w in lw) / len(lw)
            if line_text.strip():
                elements.append({
                    "text":      line_text.strip(),
                    "font_size": round(avg_size, 1),
                    "is_header": avg_size > median_size * 1.15,
                    "y":         y_key,
                })
    except Exception:
        pass
    return elements


# ------------------------------------------------
# TABLE KNOWLEDGE GRAPH
# Implements: "Table Knowledge Graph" node in architecture
# Maps: BalanceSheet → 2023 Revenue → Net Income
# ------------------------------------------------

FINANCIAL_ENTITY_PATTERNS = {
    "total_revenue":     r"total\s+revenue[s]?",
    "net_revenue":       r"net\s+revenue[s]?",
    "gross_profit":      r"gross\s+profit",
    "operating_income":  r"(?:income|loss)\s+from\s+operations|operating\s+(?:income|loss)",
    "net_income":        r"net\s+(?:income|loss)",
    "ebitda":            r"\bebitda\b",
    "total_assets":      r"total\s+assets",
    "total_liabilities": r"total\s+liabilities",
    "cash":              r"cash\s+and\s+cash\s+equivalents",
    "free_cash_flow":    r"free\s+cash\s+flow",
    "eps_diluted":       r"diluted\s+(?:net\s+)?(?:earnings|loss)\s+per\s+share",
    "gross_margin":      r"gross\s+margin",
    "operating_margin":  r"operating\s+margin",
}

def build_table_knowledge_graph(tables_with_layout, fiscal_year, ticker, page_index):
    kg_nodes = []
    for tbl in tables_with_layout:
        for entity_name, pattern in FINANCIAL_ENTITY_PATTERNS.items():
            if not re.search(pattern, tbl["markdown"], re.IGNORECASE):
                continue
            for line in tbl["markdown"].split("\n"):
                if re.search(pattern, line, re.IGNORECASE):
                    values = re.findall(r"[\(]?\$?\s*\d[\d,\.]+[\)]?", line)
                    if values:
                        kg_nodes.append({
                            "entity":      entity_name,
                            "raw_values":  values[:4],
                            "ticker":      ticker,
                            "fiscal_year": fiscal_year,
                            "page_index":  page_index,
                            "table_index": tbl["table_index"],
                            "col_headers": tbl["headers"][:4],
                            "bbox":        tbl.get("bbox"),
                        })
                    break
    return kg_nodes


# ------------------------------------------------
# PDF PROCESSOR
# ------------------------------------------------

def process_pdf(pdf_path):
    document_id  = Path(pdf_path).stem
    pages_output = []

    with pdfplumber.open(pdf_path) as pdf:
        full_text_sample = ""

        for i, page in enumerate(pdf.pages):
            page_text         = page.extract_text() or ""
            full_text_sample += page_text[:2000]

            tables_with_layout = extract_tables_with_layout(page)
            layout_elements    = detect_headers_and_paragraphs(page)
            headers_found      = [e["text"] for e in layout_elements if e["is_header"]][:10]

            page_md = md(page_text)
            if tables_with_layout:
                page_md += "\n\n" + "\n\n".join(t["markdown"] for t in tables_with_layout)

            financial_tags = detect_financial_tags(page_text)
            key_numbers    = extract_key_numbers(page_text)

            pages_output.append({
                "Document_ID":      document_id,
                "Page_Index":       i + 1,
                "Content_MD":       page_md,
                "Financial_Tags":   financial_tags,
                "Key_Numbers":      key_numbers,
                "Char_Count":       len(page_text),
                "Tables":           tables_with_layout,
                "Headers_Detected": headers_found,
                "Layout_Elements":  layout_elements[:30],
            })

    # Metadata
    doc_type    = detect_document_type(document_id, full_text_sample)
    ticker      = extract_ticker(full_text_sample)
    fiscal_year = extract_fiscal_year(full_text_sample)
    quarter     = extract_quarter(document_id, full_text_sample)

    all_kg_nodes = []
    for page in pages_output:
        kg = build_table_knowledge_graph(
            page["Tables"], fiscal_year, ticker, page["Page_Index"]
        )
        page["KG_Nodes"]      = kg
        page["Ticker"]        = ticker
        page["Fiscal_Year"]   = fiscal_year
        page["Document_Type"] = doc_type
        page["Quarter"]       = quarter
        page["Tags"]          = list(set(
            ["SEC_Filing", doc_type, ticker, fiscal_year] + page["Financial_Tags"]
        ))
        all_kg_nodes.extend(kg)

    print(f"  → {len(pages_output)} pages | {ticker} | {doc_type} | {fiscal_year} | {len(all_kg_nodes)} KG nodes")
    return pages_output, all_kg_nodes


# ------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------

def run_pipeline():
    pdf_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"No PDFs found in {INPUT_FOLDER}")
        return

    all_documents = []
    all_kg_nodes  = []

    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file}")
        try:
            pages, kg = process_pdf(INPUT_FOLDER / pdf_file)
            all_documents.extend(pages)
            all_kg_nodes.extend(kg)
        except Exception as e:
            print(f"  ERROR: {e}")

    pages_file = OUTPUT_FOLDER / "processed_sec_filings.json"
    kg_file    = OUTPUT_FOLDER / "table_knowledge_graph.json"

    with open(pages_file, "w") as f:
        json.dump(all_documents, f, indent=2)
    with open(kg_file, "w") as f:
        json.dump(all_kg_nodes, f, indent=2)

    print(f"\nDone — {len(all_documents)} pages | {len(all_kg_nodes)} KG nodes")
    print(f"Pages: {pages_file}")
    print(f"KG:    {kg_file}")

if __name__ == "__main__":
    run_pipeline()
