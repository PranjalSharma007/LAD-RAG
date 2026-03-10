"""
build_tree.py — LAD-RAG Hierarchical Index Builder
Builds: Doc ID → Section (Item 7) → Page Index → Sub-sections
Also builds structured Table Knowledge Graph index for fast financial lookup.
Architecture: PageIndex Tree & Hierarchical Index + Table Knowledge Graph
"""

import json
import re
from collections import defaultdict
from transformers import pipeline
from tqdm import tqdm


# ------------------------------------------------
# FILE PATHS
# ------------------------------------------------

INPUT_FILE  = "processed_documents/processed_sec_filings.json"
KG_FILE     = "processed_documents/table_knowledge_graph.json"
OUTPUT_FILE = "processed_documents/pageindex_tree.json"


# ------------------------------------------------
# LOAD SUMMARIZATION MODEL
# ------------------------------------------------

print("Loading summarization model...")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


# ------------------------------------------------
# SECTION DETECTION — 25 patterns, most-specific first
# ------------------------------------------------

SECTION_PATTERNS = [
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
    ("Quarterly Financial Data",
        r"(?:three|six|nine)\s+months\s+ended"),
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
        r"united\s+states\s+securities\s+and\s+exchange\s+commission|\bform\s+10-[kq]\b"),
]

def detect_section(text):
    for name, pat in SECTION_PATTERNS:
        if re.search(pat, text[:2000], re.IGNORECASE | re.MULTILINE):
            return name
    return "Other"


# ------------------------------------------------
# FINANCIAL-AWARE SUMMARIZATION
# ------------------------------------------------

_FIN_LEAD = re.compile(
    r"(?:revenue|net income|net loss|operating income|gross profit|"
    r"cash flow|earnings per share|total assets|ebitda)[^\n.]{5,150}[.\n]",
    re.IGNORECASE
)

def summarize_page(text):
    text = text.strip()
    if len(text) < 150:
        return text[:100]
    lead  = ""
    m     = _FIN_LEAD.search(text)
    if m:
        lead = m.group(0).strip()
    bart_input = (lead + " " + text[:900]).strip()[:1000]
    try:
        return summarizer(bart_input, max_length=60, min_length=20, do_sample=False)[0]["summary_text"]
    except Exception:
        return text[:200]


# ------------------------------------------------
# BUILD HIERARCHICAL TREE
# Structure: root → document → section → page
# ------------------------------------------------

def build_tree():
    print("Loading processed pages...")
    with open(INPUT_FILE) as f:
        pages = json.load(f)

    # Load KG if available
    kg_nodes = []
    try:
        with open(KG_FILE) as f:
            kg_nodes = json.load(f)
        print(f"Loaded {len(kg_nodes)} KG nodes")
    except FileNotFoundError:
        print("No KG file found — skipping KG index")

    # ── Section tree ──────────────────────────────
    section_tree = defaultdict(list)

    # ── Hierarchical tree: doc_id → section → pages ──
    hierarchical = defaultdict(lambda: defaultdict(list))

    print("Building PageIndex Tree...")
    for page in tqdm(pages, desc="Processing pages"):
        text    = page.get("Content_MD", "")
        section = detect_section(text)
        summary = summarize_page(text)

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
            "headers":       page.get("Headers_Detected", []),
            "has_tables":    len(page.get("Tables", [])) > 0,
            "table_count":   len(page.get("Tables", [])),
            "kg_node_count": len(page.get("KG_Nodes", [])),
            "summary":       summary,
            "section":       section,
        }

        section_tree[section].append(node)
        hierarchical[page["Document_ID"]][section].append(node)

    # ── Table Knowledge Graph index ──────────────
    # Group by entity for fast O(1) lookup
    kg_index = defaultdict(list)
    for node in kg_nodes:
        kg_index[node["entity"]].append(node)

    # ── Build citation map: page → table reference ──
    citation_map = {}
    for page in pages:
        for tbl in page.get("Tables", []):
            cit_key = f"Page {page['Page_Index']} Table {tbl['table_index']+1}"
            citation_map[cit_key] = {
                "page":        page["Page_Index"],
                "table":       tbl["table_index"] + 1,
                "headers":     tbl.get("headers", [])[:4],
                "bbox":        tbl.get("bbox"),
                "document_id": page["Document_ID"],
            }

    document_tree = {
        "root":     "SEC Filing",
        "sections": dict(section_tree),
        "hierarchical_index": {
            doc_id: dict(sections)
            for doc_id, sections in hierarchical.items()
        },
        "table_knowledge_graph": {
            entity: nodes
            for entity, nodes in kg_index.items()
        },
        "citation_map": citation_map,
        "meta": {
            "total_pages":   len(pages),
            "section_count": len(section_tree),
            "kg_entities":   len(kg_index),
            "citations":     len(citation_map),
        }
    }

    print("Saving tree...")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(document_tree, f, indent=2)

    print(f"\nPageIndex Tree built successfully")
    print(f"  Sections:    {list(section_tree.keys())}")
    print(f"  KG entities: {list(kg_index.keys())}")
    print(f"  Citations:   {len(citation_map)}")
    print(f"  Saved to:    {OUTPUT_FILE}")


if __name__ == "__main__":
    build_tree()
