"""
navigator.py — Enhanced LAD-RAG Navigator
Retrieves relevant SEC filing pages and synthesizes analyst-grade answers
via Claude API with financial-focused prompting.
"""

import json
import os
import re
from sentence_transformers import SentenceTransformer, util
import anthropic


# ------------------------------------------------
# FILE PATHS
# ------------------------------------------------

TREE_FILE  = "processed_documents/pageindex_tree.json"
PAGES_FILE = "processed_documents/processed_sec_filings.json"


# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------

print("Loading PageIndex Tree...")
with open(TREE_FILE) as f:
    tree = json.load(f)

print("Loading Page Text...")
with open(PAGES_FILE) as f:
    pages = json.load(f)

# Build fast page lookup by index
page_by_index = {p["Page_Index"]: p for p in pages}


# ------------------------------------------------
# LOAD EMBEDDING MODEL
# ------------------------------------------------

print("Loading semantic model...")
model = SentenceTransformer("all-MiniLM-L6-v2")


# ------------------------------------------------
# BUILD RICH SUMMARY INDEX
# ------------------------------------------------

summary_texts  = []
page_lookup    = []
page_meta      = []   # store section + financial tags per entry

for section, nodes in tree["sections"].items():
    for node in nodes:

        # Richer embedding text — includes financial tags and key numbers
        fin_tags     = " ".join(node.get("financial_tags", []))
        key_nums_raw = node.get("key_numbers", {})
        dollar_str   = " ".join(key_nums_raw.get("dollar_amounts", []))
        pct_str      = " ".join(key_nums_raw.get("percentages", []))
        yoy_str      = " ".join(key_nums_raw.get("yoy_changes_pct", []))

        text = (
            f"Section: {section} | "
            f"Filing: {node['filing_type']} {node.get('fiscal_year','')} "
            f"{node.get('quarter') or ''} | "
            f"Ticker: {node['ticker']} | "
            f"Topics: {fin_tags} | "
            f"Figures: {dollar_str} {pct_str} {yoy_str} | "
            f"Summary: {node['summary']}"
        )

        summary_texts.append(text)
        page_lookup.append(node["page_index"])
        page_meta.append({
            "section":       section,
            "financial_tags": node.get("financial_tags", []),
            "char_count":    node.get("char_count", 0),
        })

print(f"Embedding {len(summary_texts)} page summaries...")
summary_embeddings = model.encode(summary_texts, convert_to_tensor=True)
print("Ready.\n")


# ------------------------------------------------
# FINANCIAL KEYWORD BOOST MAP
# ------------------------------------------------

KEYWORD_BOOSTS = {
    # Revenue / growth
    "revenue":          0.20,
    "net revenue":      0.20,
    "total revenue":    0.20,
    "growth":           0.15,
    "yoy":              0.15,
    "year over year":   0.15,
    # Profitability
    "net income":       0.20,
    "net loss":         0.20,
    "gross profit":     0.18,
    "gross margin":     0.18,
    "operating income": 0.18,
    "ebitda":           0.18,
    "operating margin": 0.15,
    # Balance sheet
    "total assets":     0.15,
    "liabilities":      0.15,
    "equity":           0.12,
    "cash":             0.12,
    # Cash flow
    "free cash flow":   0.18,
    "cash flow":        0.15,
    "capex":            0.12,
    # EPS
    "eps":              0.15,
    "earnings per share": 0.15,
    "diluted":          0.10,
    # Guidance
    "guidance":         0.15,
    "outlook":          0.15,
    "forecast":         0.15,
    # General financial
    "financial":        0.10,
    "statement":        0.08,
    "balance sheet":    0.12,
}


# ------------------------------------------------
# NAVIGATOR — semantic + keyword boost + length filter
# ------------------------------------------------

def navigate(query: str, top_k: int = 5) -> list[int]:

    query_lower = query.lower()
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, summary_embeddings)[0].tolist()

    boosted = []
    for i, score in enumerate(scores):
        text = summary_texts[i].lower()
        boost = 0.0

        for keyword, kboost in KEYWORD_BOOSTS.items():
            if keyword in query_lower and keyword in text:
                boost += kboost

        # Penalise near-empty pages (cover, TOC, etc.)
        if page_meta[i]["char_count"] < 300:
            boost -= 0.25

        # Bonus for pages that are content-rich
        if page_meta[i]["char_count"] > 1500:
            boost += 0.05

        boosted.append(score + boost)

    ranked = sorted(range(len(boosted)), key=lambda i: boosted[i], reverse=True)

    # Deduplicate by section — prefer variety
    seen_sections = set()
    selected = []
    remainder = []

    for idx in ranked:
        sec = page_meta[idx]["section"]
        if sec not in seen_sections:
            selected.append(page_lookup[idx])
            seen_sections.add(sec)
        else:
            remainder.append(page_lookup[idx])
        if len(selected) >= top_k:
            break

    # Fill remaining slots from remainder if needed
    for idx in remainder:
        if len(selected) >= top_k:
            break
        selected.append(page_lookup[idx])

    return selected


# ------------------------------------------------
# CONTEXT EXTRACTION
# ------------------------------------------------

def retrieve_pages(page_indices: list[int]) -> str:

    chunks = []

    for idx in page_indices:
        page = page_by_index.get(idx)
        if page:
            header = (
                f"--- Page {idx} | "
                f"{page.get('Document_Type','')} "
                f"{page.get('Fiscal_Year','')} "
                f"{page.get('Quarter') or ''} | "
                f"Ticker: {page.get('Ticker','')} ---"
            )
            chunks.append(header + "\n" + page["Content_MD"])

    return "\n\n".join(chunks)


# ------------------------------------------------
# FINANCIAL ANALYST SYSTEM PROMPT
# ------------------------------------------------

SYSTEM_PROMPT = """You are a senior equity research analyst. Your job is to answer questions
about SEC filings (10-K, 10-Q) with the precision and format expected by institutional investors.

RESPONSE FORMAT RULES:
1. Lead with the direct numerical answer (dollar figure, %, or ratio) in the first sentence.
2. Provide year-over-year or quarter-over-quarter comparison whenever the data is available.
3. Highlight key drivers of the change (volume, price, mix, geography, product segment).
4. Flag any risks, one-time items, or non-GAAP adjustments that affect comparability.
5. If the exact figure is not in the context, say so clearly and state what CAN be inferred.
6. Use concise analyst prose — no filler phrases, no unnecessary hedging.
7. Structure your answer with these sections when relevant:
   • KEY FIGURE — the headline number
   • BREAKDOWN — segment / product / geo detail
   • DRIVERS — what caused the change
   • COMPARABLES — prior period or peer context if available
   • WATCH ITEMS — risks, non-recurring items, guidance

DO NOT fabricate numbers. If a number is not in the provided context, say "Not disclosed in retrieved pages."
"""


# ------------------------------------------------
# LLM ANSWER SYNTHESIS — via Claude API
# ------------------------------------------------

def synthesize_answer(query: str, context: str) -> str:

    client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from env

    user_message = (
        f"ANALYST QUERY: {query}\n\n"
        f"FILING CONTEXT (extracted pages):\n\n"
        f"{context[:12000]}"          # stay within context window
    )

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1200,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    return response.content[0].text


# ------------------------------------------------
# FULL PIPELINE
# ------------------------------------------------

def answer_query(query: str) -> str:

    print(f"\n{'='*60}")
    print(f"QUERY: {query}")

    page_indices = navigate(query)
    print(f"Navigator → pages: {page_indices}")

    context = retrieve_pages(page_indices)

    print("\n--- Context preview (first 600 chars) ---")
    print(context[:600])
    print("---\n")

    print("Synthesizing analyst answer...")
    answer = synthesize_answer(query, context)

    print("\n" + "="*60)
    print("ANALYST ANSWER:")
    print("="*60)
    print(answer)
    print("="*60)

    return answer


# ------------------------------------------------
# INTERACTIVE LOOP
# ------------------------------------------------

EXAMPLE_QUERIES = [
    "What was the total revenue in FY2024 and how did it grow YoY?",
    "What is the gross margin trend over the last 4 quarters?",
    "What are the key risk factors that could impact revenue?",
    "What is the free cash flow and how does it compare to net income?",
    "What guidance did management provide for the next fiscal year?",
]

if __name__ == "__main__":

    print("\nExample queries you can ask:")
    for i, q in enumerate(EXAMPLE_QUERIES, 1):
        print(f"  {i}. {q}")

    print("\nType 'exit' to quit.\n")

    while True:
        query = input("Analyst Query > ").strip()

        if not query:
            continue

        if query.lower() in ("exit", "quit"):
            break

        # Allow shortcut: type a number to use an example query
        if query.isdigit() and 1 <= int(query) <= len(EXAMPLE_QUERIES):
            query = EXAMPLE_QUERIES[int(query) - 1]
            print(f"Using example: {query}")

        try:
            answer_query(query)
        except anthropic.APIError as e:
            print(f"\nAPI error: {e}")
        except Exception as e:
            print(f"\nError: {e}")
