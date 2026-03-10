import json
import re
from collections import defaultdict
from transformers import pipeline
from tqdm import tqdm


# ------------------------------------------------
# FILE PATHS
# ------------------------------------------------

INPUT_FILE  = "processed_documents/processed_sec_filings.json"
OUTPUT_FILE = "processed_documents/pageindex_tree.json"


# ------------------------------------------------
# LOAD SUMMARIZATION MODEL
# ------------------------------------------------

print("Loading summarization model...")

summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)


# ------------------------------------------------
# ENHANCED SECTION DETECTION
# ------------------------------------------------

# Ordered from most specific to least — first match wins
SECTION_PATTERNS = [
    # 10-K standard items
    ("Item 1A Risk Factors",              r"Item\s+1A[\.\s]"),
    ("Item 1B Unresolved Staff Comments", r"Item\s+1B[\.\s]"),
    ("Item 1 Business",                   r"Item\s+1[\.\s](?!A|B)"),
    ("Item 2 Properties",                 r"Item\s+2[\.\s]"),
    ("Item 3 Legal Proceedings",          r"Item\s+3[\.\s]"),
    ("Item 4 Mine Safety",                r"Item\s+4[\.\s]"),
    ("Item 5 Market Info",                r"Item\s+5[\.\s]"),
    ("Item 6 Selected Data",              r"Item\s+6[\.\s]"),
    ("Item 7A Quantitative Risk",         r"Item\s+7A[\.\s]"),
    ("Item 7 MD&A",                       r"Item\s+7[\.\s](?!A)"),
    ("Item 8 Financial Statements",       r"Item\s+8[\.\s]"),
    ("Item 9A Controls",                  r"Item\s+9A[\.\s]"),
    ("Item 9 Disagreements",              r"Item\s+9[\.\s](?!A)"),

    # Financial statement types (common table headers)
    ("Consolidated Income Statement",
        r"consolidated\s+statements?\s+of\s+(?:operations|income|earnings)"),
    ("Consolidated Balance Sheet",
        r"consolidated\s+balance\s+sheets?"),
    ("Consolidated Cash Flow",
        r"consolidated\s+statements?\s+of\s+cash\s+flows?"),
    ("Consolidated Equity Statement",
        r"consolidated\s+statements?\s+of\s+(?:stockholders|shareholders).{0,20}equity"),
    ("Notes to Financial Statements",
        r"notes?\s+to\s+(?:the\s+)?(?:consolidated\s+)?financial\s+statements?"),

    # 10-Q specific
    ("Quarterly Financial Data",
        r"(?:three|six|nine)\s+months\s+ended"),

    # Revenue / earnings highlights
    ("Revenue",
        r"\btotal\s+revenue\b|\bnet\s+revenue\b|\brevenue\s+recognition\b"),
    ("Earnings Per Share",
        r"\bearnings\s+per\s+(?:common\s+)?share\b|\bdiluted\s+eps\b"),
    ("Segment Information",
        r"\bsegment\b.{0,30}\breport(?:ing|ed)\b|\boperating\s+segment\b"),
    ("Guidance / Outlook",
        r"\boutlook\b|\bguidance\b|\bforward.{0,10}looking\b"),
    ("Table of Contents",
        r"^table\s+of\s+contents\b|\bindex\s+to\s+financial"),
    ("Cover Page",
        r"united\s+states\s+securities\s+and\s+exchange\s+commission"
        r"|\bform\s+10-[kq]\b"),
]


def detect_section(text):
    for section_name, pattern in SECTION_PATTERNS:
        if re.search(pattern, text[:2000], re.IGNORECASE | re.MULTILINE):
            return section_name
    return "Other"


# ------------------------------------------------
# PAGE SUMMARIZATION — FINANCIAL-AWARE
# ------------------------------------------------

# Extract key financial sentence to prepend to BART input
_FINANCIAL_LEAD = re.compile(
    r"(?:revenue|net income|net loss|operating income|gross profit|"
    r"cash flow|earnings per share|total assets|ebitda)[^\n.]{5,120}[.\n]",
    re.IGNORECASE
)

def extract_financial_lead(text):
    match = _FINANCIAL_LEAD.search(text)
    if match:
        return match.group(0).strip()
    return ""


def summarize_page(text):

    text = text.strip()

    if len(text) < 150:
        return text[:100]

    # Try to surface a financial lead sentence for context
    lead = extract_financial_lead(text)

    # BART input: financial lead + first chunk
    bart_input = (lead + " " + text[:900]).strip()[:1000]

    try:
        result = summarizer(
            bart_input,
            max_length=60,   # longer for financial context
            min_length=20,
            do_sample=False
        )[0]["summary_text"]
    except Exception:
        result = text[:200]

    return result


# ------------------------------------------------
# BUILD PAGE INDEX TREE
# ------------------------------------------------

def build_tree():

    print("Loading processed pages...")

    with open(INPUT_FILE) as f:
        pages = json.load(f)

    tree = defaultdict(list)

    print("Building PageIndex Tree...")

    for page in tqdm(pages, desc="Processing pages"):

        text = page.get("Content_MD", "")

        section  = detect_section(text)
        summary  = summarize_page(text)

        node = {
            "page_index":    page["Page_Index"],
            "document_id":   page["Document_ID"],
            "ticker":        page.get("Ticker", "UNKNOWN"),
            "filing_type":   page.get("Document_Type", "UNKNOWN"),
            "fiscal_year":   page.get("Fiscal_Year", "UNKNOWN"),
            "quarter":       page.get("Quarter"),
            "financial_tags":page.get("Financial_Tags", []),
            "key_numbers":   page.get("Key_Numbers", {}),
            "char_count":    page.get("Char_Count", 0),
            "summary":       summary,
        }

        tree[section].append(node)

    document_tree = {
        "root":     "SEC Filing",
        "sections": dict(tree),
        "meta": {
            "total_pages":    len(pages),
            "section_count":  len(tree),
        }
    }

    print("Saving tree...")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(document_tree, f, indent=2)

    print("\nPageIndex Tree built successfully")
    print(f"  Sections detected: {list(tree.keys())}")
    print(f"  Saved to: {OUTPUT_FILE}")


# ------------------------------------------------
# ENTRY POINT
# ------------------------------------------------

if __name__ == "__main__":
    build_tree()
