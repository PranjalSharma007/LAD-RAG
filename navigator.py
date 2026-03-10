"""
navigator.py — LAD-RAG Full Architecture Navigator
Implements:
  1. Multi-step Query Engine (semantic + finance dict expansion + structured table search)
  2. Retriever (top-K pages + preserved tables)
  3. Reranker (finance-term late interaction scoring, ColBERT-style)
  4. Table Knowledge Graph lookup
  5. LLM synthesis via Groq (llama-3.3-70b-versatile)
  6. Python Calculator Tool for financial math
  7. Citations: "See Table 2 on Page 45"
"""

import json
import os
import re
import ast
import math
import operator
from groq import Groq
from sentence_transformers import SentenceTransformer, util


# ─────────────────────────────────────────────────
# API KEY
# ─────────────────────────────────────────────────
os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY_HERE"   # ← paste your key


# ─────────────────────────────────────────────────
# FILE PATHS
# ─────────────────────────────────────────────────

TREE_FILE  = "processed_documents/pageindex_tree.json"
PAGES_FILE = "processed_documents/processed_sec_filings.json"


# ─────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────

print("Loading PageIndex Tree...")
with open(TREE_FILE) as f:
    tree = json.load(f)

print("Loading Page Text...")
with open(PAGES_FILE) as f:
    pages = json.load(f)

page_by_index  = {p["Page_Index"]: p for p in pages}
kg_index       = tree.get("table_knowledge_graph", {})
citation_map   = tree.get("citation_map", {})
hier_index     = tree.get("hierarchical_index", {})


# ─────────────────────────────────────────────────
# LOAD EMBEDDING MODEL
# ─────────────────────────────────────────────────

print("Loading semantic model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# ─────────────────────────────────────────────────
# BUILD RICH SUMMARY INDEX
# ─────────────────────────────────────────────────

summary_texts = []
page_lookup   = []
page_meta     = []

for section, nodes in tree["sections"].items():
    for node in nodes:
        fin_tags   = " ".join(node.get("financial_tags", []))
        kn         = node.get("key_numbers", {})
        dollar_str = " ".join(kn.get("dollar_amounts", []))
        pct_str    = " ".join(kn.get("percentages", []))
        yoy_str    = " ".join(kn.get("yoy_changes_pct", []))
        headers    = " ".join(node.get("headers", []))

        text = (
            f"Section: {section} | "
            f"Filing: {node['filing_type']} {node.get('fiscal_year','')} "
            f"{node.get('quarter') or ''} | Ticker: {node['ticker']} | "
            f"Topics: {fin_tags} | Figures: {dollar_str} {pct_str} {yoy_str} | "
            f"Headers: {headers} | Summary: {node['summary']}"
        )
        summary_texts.append(text)
        page_lookup.append(node["page_index"])
        page_meta.append({
            "section":        section,
            "financial_tags": node.get("financial_tags", []),
            "char_count":     node.get("char_count", 0),
            "has_tables":     node.get("has_tables", False),
            "table_count":    node.get("table_count", 0),
        })

print(f"Embedding {len(summary_texts)} page summaries...")
summary_embeddings = embedder.encode(summary_texts, convert_to_tensor=True)
print("Ready.\n")


# ─────────────────────────────────────────────────
# FINANCE DICTIONARY — Query Expansion
# Implements: "Semantic Query Expansion (Finance Dict.)"
# ─────────────────────────────────────────────────

FINANCE_DICT = {
    "revenue":         ["net revenue", "total revenue", "sales", "top line"],
    "profit":          ["net income", "net earnings", "bottom line", "net profit"],
    "margin":          ["gross margin", "operating margin", "net margin", "ebitda margin"],
    "growth":          ["year over year", "yoy", "quarter over quarter", "qoq", "increase", "decrease"],
    "cash":            ["free cash flow", "fcf", "operating cash flow", "cash from operations"],
    "debt":            ["long term debt", "borrowings", "credit facility", "leverage"],
    "expenses":        ["operating expenses", "opex", "cost of revenue", "sg&a", "r&d"],
    "guidance":        ["outlook", "forecast", "full year", "next quarter", "q4 guidance"],
    "earnings":        ["eps", "earnings per share", "diluted eps", "basic eps"],
    "assets":          ["total assets", "current assets", "non-current assets", "balance sheet"],
    "risk":            ["risk factors", "material weakness", "litigation", "regulatory"],
    "segment":         ["business segment", "product line", "geographic segment", "regional"],
    "operating income":["income from operations", "operating profit", "ebit"],
    "capex":           ["capital expenditure", "capital spending", "property and equipment"],
}

def expand_query(query: str) -> str:
    """Add finance dictionary synonyms to the query for broader semantic coverage."""
    q_lower    = query.lower()
    expansions = []
    for term, synonyms in FINANCE_DICT.items():
        if term in q_lower:
            expansions.extend(synonyms)
        for syn in synonyms:
            if syn in q_lower:
                expansions.append(term)
                break
    if expansions:
        expanded = query + " " + " ".join(set(expansions))
        return expanded
    return query


# ─────────────────────────────────────────────────
# KEYWORD BOOST MAP
# ─────────────────────────────────────────────────

KEYWORD_BOOSTS = {
    "revenue": 0.20, "net revenue": 0.20, "total revenue": 0.20,
    "growth": 0.15,  "yoy": 0.15,         "year over year": 0.15,
    "net income": 0.20, "net loss": 0.20, "gross profit": 0.18,
    "gross margin": 0.18, "operating income": 0.18, "ebitda": 0.18,
    "operating margin": 0.15, "total assets": 0.15, "liabilities": 0.15,
    "equity": 0.12, "cash": 0.12, "free cash flow": 0.18,
    "cash flow": 0.15, "capex": 0.12, "eps": 0.15,
    "earnings per share": 0.15, "diluted": 0.10,
    "guidance": 0.15, "outlook": 0.15, "forecast": 0.15,
    "financial": 0.10, "balance sheet": 0.12, "segment": 0.10,
}


# ─────────────────────────────────────────────────
# STRUCTURED TABLE SEARCH
# Implements: "Structured Table Search Query"
# ─────────────────────────────────────────────────

def search_knowledge_graph(query: str) -> list[dict]:
    """
    Directly query the Table Knowledge Graph for financial entities.
    Returns matching KG nodes with page citations.
    """
    q_lower  = query.lower()
    matches  = []
    entity_map = {
        "revenue":          ["total_revenue", "net_revenue"],
        "profit":           ["net_income", "gross_profit"],
        "income":           ["net_income", "operating_income"],
        "loss":             ["net_income"],
        "margin":           ["gross_margin", "operating_margin"],
        "assets":           ["total_assets"],
        "liabilities":      ["total_liabilities"],
        "cash":             ["cash", "free_cash_flow"],
        "eps":              ["eps_diluted"],
        "earnings per share": ["eps_diluted"],
        "ebitda":           ["ebitda"],
        "capex":            ["capex"],
        "operating income": ["operating_income"],
        "free cash flow":   ["free_cash_flow"],
    }

    entities_to_search = set()
    for keyword, entities in entity_map.items():
        if keyword in q_lower:
            entities_to_search.update(entities)

    for entity in entities_to_search:
        nodes = kg_index.get(entity, [])
        matches.extend(nodes[:3])   # top 3 per entity

    return matches


# ─────────────────────────────────────────────────
# RERANKER — Finance-Tuned Late Interaction
# Implements: "Reranker (Finance-Tuned ColBERT)"
# Late interaction: score each term in query against page text
# ─────────────────────────────────────────────────

FINANCIAL_RERANK_TERMS = [
    "revenue", "net income", "gross profit", "operating income",
    "ebitda", "cash flow", "earnings per share", "total assets",
    "margin", "growth", "guidance", "segment", "yoy", "qoq",
    "billion", "million", "percent", "increase", "decrease",
]

def rerank_pages(query: str, candidate_indices: list[int], top_k: int = 5) -> list[int]:
    """
    Late interaction reranking:
    Score pages by how many financial query terms appear in their full text.
    """
    q_lower  = query.lower()
    q_terms  = set(q_lower.split())

    # Add financial terms that appear in the query
    active_fin_terms = [t for t in FINANCIAL_RERANK_TERMS if t in q_lower]

    scored = []
    for page_idx in candidate_indices:
        page = page_by_index.get(page_idx)
        if not page:
            scored.append((page_idx, 0))
            continue

        page_text = page.get("Content_MD", "").lower()
        score     = 0.0

        # Term overlap score (late interaction)
        for term in q_terms:
            if len(term) > 3 and term in page_text:
                score += 0.5

        # Financial term boost (key financial vocabulary match)
        for fin_term in active_fin_terms:
            count  = page_text.count(fin_term)
            score += min(count * 0.3, 2.0)   # cap per term

        # Table bonus — tables contain structured numbers
        if page.get("Tables"):
            score += len(page["Tables"]) * 0.4

        # KG node bonus — page has extracted financial entities
        if page.get("KG_Nodes"):
            score += len(page["KG_Nodes"]) * 0.5

        scored.append((page_idx, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in scored[:top_k]]


# ─────────────────────────────────────────────────
# MULTI-STEP NAVIGATOR
# Implements: "Multi-step Query Engine"
# Step 1: Semantic search with query expansion
# Step 2: KG structured search
# Step 3: Reranker
# ─────────────────────────────────────────────────

def navigate(query: str, top_k: int = 5) -> tuple[list[int], list[dict]]:
    """
    Returns: (page_indices, kg_matches)
    """
    # ── Step 1: Semantic search with finance dict expansion ──
    expanded        = expand_query(query)
    q_lower         = expanded.lower()
    query_embedding = embedder.encode(expanded, convert_to_tensor=True)
    scores          = util.cos_sim(query_embedding, summary_embeddings)[0].tolist()

    # Apply keyword boosts
    boosted = []
    for i, score in enumerate(scores):
        text  = summary_texts[i].lower()
        boost = 0.0
        for kw, kb in KEYWORD_BOOSTS.items():
            if kw in q_lower and kw in text:
                boost += kb
        if page_meta[i]["char_count"] < 300:
            boost -= 0.25
        if page_meta[i]["has_tables"]:
            boost += page_meta[i]["table_count"] * 0.05
        if page_meta[i]["char_count"] > 1500:
            boost += 0.05
        boosted.append(score + boost)

    ranked = sorted(range(len(boosted)), key=lambda i: boosted[i], reverse=True)

    # Get diverse candidate pool (3x top_k)
    seen_sections = set()
    candidates    = []
    for idx in ranked:
        sec = page_meta[idx]["section"]
        if sec not in seen_sections or len(candidates) < top_k * 2:
            candidates.append(page_lookup[idx])
            seen_sections.add(sec)
        if len(candidates) >= top_k * 3:
            break

    # ── Step 2: Structured KG search ──
    kg_matches = search_knowledge_graph(query)

    # Add KG page indices to candidates
    for kg_node in kg_matches:
        pi = kg_node.get("page_index")
        if pi and pi not in candidates:
            candidates.append(pi)

    # ── Step 3: Rerank ──
    final_pages = rerank_pages(query, candidates, top_k=top_k)

    return final_pages, kg_matches


# ─────────────────────────────────────────────────
# CONTEXT EXTRACTION WITH CITATIONS
# Implements: "See Table 2 on Page 45" citations
# ─────────────────────────────────────────────────

def retrieve_pages(page_indices: list[int]) -> tuple[str, list[str]]:
    """Returns (context_text, citation_list)"""
    chunks    = []
    citations = []

    for idx in page_indices:
        page = page_by_index.get(idx)
        if not page:
            continue

        # Build citation references for tables on this page
        page_citations = []
        for tbl in page.get("Tables", []):
            tbl_num  = tbl["table_index"] + 1
            cite_ref = f"Table {tbl_num} on Page {idx}"
            page_citations.append(cite_ref)
            citations.append({
                "ref":      cite_ref,
                "page":     idx,
                "table":    tbl_num,
                "headers":  tbl.get("headers", [])[:4],
                "document": page.get("Document_ID", ""),
            })

        cite_str = " [" + ", ".join(page_citations) + "]" if page_citations else ""

        header = (
            f"--- Page {idx}{cite_str} | "
            f"{page.get('Document_Type','')} {page.get('Fiscal_Year','')} "
            f"{page.get('Quarter') or ''} | Ticker: {page.get('Ticker','')} ---"
        )
        chunks.append(header + "\n" + page["Content_MD"])

    return "\n\n".join(chunks), citations


# ─────────────────────────────────────────────────
# PYTHON CALCULATOR TOOL
# Implements: "Python Calculator/Interpreter Tool"
# Used by LLM for precise financial math
# ─────────────────────────────────────────────────

SAFE_OPS = {
    ast.Add:  operator.add,
    ast.Sub:  operator.sub,
    ast.Mult: operator.mul,
    ast.Div:  operator.truediv,
    ast.Pow:  operator.pow,
    ast.USub: operator.neg,
}

def safe_eval(expr: str) -> float:
    """Safe math expression evaluator — no exec, no eval."""
    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            return SAFE_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            return SAFE_OPS[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.Call):
            fn_map = {
                "round": round, "abs": abs,
                "sqrt": math.sqrt, "log": math.log,
            }
            func = fn_map.get(node.func.id if isinstance(node.func, ast.Name) else "")
            if func:
                args = [_eval(a) for a in node.args]
                return func(*args)
        raise ValueError(f"Unsupported expression: {ast.dump(node)}")
    try:
        tree_node = ast.parse(expr.strip(), mode="eval")
        result    = _eval(tree_node.body)
        return round(float(result), 6)
    except Exception as e:
        return f"CALC_ERROR: {e}"

def extract_and_run_calculations(text: str) -> str:
    """
    Find CALC(...) blocks in LLM output and execute them.
    LLM can write: CALC(593.4 / 521.8 - 1) to compute YoY growth.
    """
    def replacer(m):
        expr   = m.group(1)
        result = safe_eval(expr)
        if isinstance(result, float):
            return f"{result:.4f}"
        return str(result)
    return re.sub(r"CALC\(([^)]+)\)", replacer, text)


# ─────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior equity research analyst with access to SEC filing data and a Python calculator tool.

RESPONSE FORMAT — always use these labeled sections:
▸ KEY FIGURE    — headline number (always lead with a dollar/% figure)
▸ BREAKDOWN     — by segment, product, geography if available
▸ DRIVERS       — what caused the change (volume, price, mix, FX, acquisitions)
▸ COMPARABLES   — prior period numbers for YoY or QoQ context
▸ WATCH ITEMS   — risks, non-recurring items, non-GAAP adjustments, guidance

CALCULATOR TOOL:
You have access to a Python calculator. Use CALC(expression) for any math.
Examples:
  - YoY growth: CALC((593.4 / 521.8) - 1)  → gives decimal, multiply by 100 for %
  - Margin:     CALC(120.5 / 593.4)         → gross margin ratio
  - Difference: CALC(593.4 - 521.8)         → absolute change

CITATION RULES:
- Always cite tables as: "See Table 2 on Page 45"
- Reference page numbers when quoting figures

STRICT RULES:
1. Lead with the exact number in sentence 1.
2. Never fabricate numbers — if not in context say "Not disclosed in retrieved pages."
3. Always show YoY or QoQ comparison when prior period data is available.
4. Flag non-GAAP items explicitly.
5. Use concise analyst prose — no filler, no hedging."""


# ─────────────────────────────────────────────────
# KG CONTEXT BUILDER
# ─────────────────────────────────────────────────

def build_kg_context(kg_matches: list[dict]) -> str:
    if not kg_matches:
        return ""
    lines = ["\n--- TABLE KNOWLEDGE GRAPH (direct entity lookup) ---"]
    for node in kg_matches[:8]:
        lines.append(
            f"Entity: {node['entity']} | "
            f"Values: {', '.join(node.get('raw_values', []))} | "
            f"Year: {node.get('fiscal_year','')} | "
            f"See Table {node.get('table_index',0)+1} on Page {node.get('page_index','?')}"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────
# GROQ LLM SYNTHESIS
# ─────────────────────────────────────────────────

def synthesize_answer(query: str, context: str, kg_context: str, citations: list[dict]) -> str:
    client = Groq()

    cite_str = ""
    if citations:
        cite_str = "\n\nAVAILABLE CITATIONS:\n" + "\n".join(
            f"  • {c['ref']} (columns: {', '.join(c['headers'][:3])})"
            for c in citations[:10]
        )

    user_message = (
        f"ANALYST QUERY: {query}\n"
        f"{cite_str}\n"
        f"{kg_context}\n\n"
        f"FILING CONTEXT:\n\n{context[:5500]}"
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1200,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
    )

    raw_answer = response.choices[0].message.content
    # Run calculator on any CALC() expressions the LLM wrote
    return extract_and_run_calculations(raw_answer)


# ─────────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────────

def answer_query(query: str) -> str:
    print(f"\n{'='*65}")
    print(f"QUERY: {query}")

    # Multi-step retrieval
    page_indices, kg_matches = navigate(query)
    print(f"Navigator  → pages: {page_indices}")
    print(f"KG matches → {len(kg_matches)} entities found")

    context, citations = retrieve_pages(page_indices)
    kg_context         = build_kg_context(kg_matches)

    print(f"Citations  → {[c['ref'] for c in citations[:5]]}")
    print("\n--- Context preview ---")
    print(context[:500])
    print("---\n")

    print("Synthesizing analyst answer (Groq / Llama-3.3-70b)...")
    answer = synthesize_answer(query, context, kg_context, citations)

    print("\n" + "="*65)
    print("ANALYST ANSWER:")
    print("="*65)
    print(answer)
    print("="*65)

    return answer


# ─────────────────────────────────────────────────
# INTERACTIVE LOOP
# ─────────────────────────────────────────────────

EXAMPLE_QUERIES = [
    "What was total revenue in FY2024 and YoY growth rate?",
    "What is the gross margin and how did it change from prior year?",
    "What is free cash flow vs net income — any divergence?",
    "What guidance did management provide for next fiscal year?",
    "What are the top 3 risk factors impacting revenue?",
    "Break down operating expenses by category YoY.",
    "What drove the change in operating income?",
    "What is diluted EPS and how does it compare to prior year?",
]

if __name__ == "__main__":
    print("\nExample queries:")
    for i, q in enumerate(EXAMPLE_QUERIES, 1):
        print(f"  {i}. {q}")
    print("\nType a number (1-8) to use an example, or type your own. 'exit' to quit.\n")

    while True:
        query = input("Analyst Query > ").strip()
        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            break
        if query.isdigit() and 1 <= int(query) <= len(EXAMPLE_QUERIES):
            query = EXAMPLE_QUERIES[int(query) - 1]
            print(f"Using: {query}")
        try:
            answer_query(query)
        except Exception as e:
            print(f"\nError: {e}")
