import json
import re
from collections import defaultdict
from transformers import pipeline
from tqdm import tqdm


# ------------------------------------------------
# FILE PATHS
# ------------------------------------------------

INPUT_FILE = "processed_documents/processed_sec_filings.json"
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
# SECTION DETECTION
# ------------------------------------------------

def detect_section(text):

    patterns = {
        "Item 1 Business": r"Item\s+1\.*\s+Business",
        "Item 1A Risk Factors": r"Item\s+1A\.*\s+Risk",
        "Item 2 Properties": r"Item\s+2\.*\s+Properties",
        "Item 7 MD&A": r"Item\s+7\.*",
        "Item 7A Quantitative Risk": r"Item\s+7A\.*",
        "Item 8 Financial Statements": r"Item\s+8\.*"
    }

    for section, pattern in patterns.items():

        if re.search(pattern, text, re.IGNORECASE):
            return section

    return "Other"


# ------------------------------------------------
# PAGE SUMMARIZATION
# ------------------------------------------------

def summarize_page(text):

    text = text.strip()

    # Skip tiny pages (headers / blank pages)
    if len(text) < 200:
        return text[:100]

    text = text[:1000]

    try:

        summary = summarizer(
            text,
            max_length=30,
            min_length=10,
            do_sample=False
        )[0]["summary_text"]

    except Exception:

        summary = text[:150]

    return summary


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

        text = page["Content_MD"]

        section = detect_section(text)

        summary = summarize_page(text)

        node = {
            "page_index": page["Page_Index"],
            "document_id": page["Document_ID"],
            "ticker": page["Ticker"],
            "filing_type": page["Document_Type"],
            "summary": summary
        }

        tree[section].append(node)

    document_tree = {
        "root": "SEC Filing",
        "sections": dict(tree)
    }

    print("Saving tree...")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(document_tree, f, indent=2)

    print("PageIndex Tree built successfully")
    print("Saved to:", OUTPUT_FILE)


# ------------------------------------------------
# ENTRY POINT
# ------------------------------------------------

if __name__ == "__main__":

    build_tree()
