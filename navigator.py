import json
from sentence_transformers import SentenceTransformer, util

# ------------------------------------------------
# FILE PATHS
# ------------------------------------------------

TREE_FILE = "processed_documents/pageindex_tree.json"
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


# ------------------------------------------------
# LOAD EMBEDDING MODEL
# ------------------------------------------------

print("Loading semantic model...")

model = SentenceTransformer("all-MiniLM-L6-v2")


# ------------------------------------------------
# BUILD PAGE SUMMARY INDEX
# ------------------------------------------------

summary_texts = []
page_lookup = []

for section, nodes in tree["sections"].items():

    for node in nodes:

        text = f"""
        Section: {section}
        Summary: {node['summary']}
        Filing: {node['filing_type']}
        """

        summary_texts.append(text)
        page_lookup.append(node["page_index"])

print("Embedding summaries...")

summary_embeddings = model.encode(summary_texts, convert_to_tensor=True)


# ------------------------------------------------
# NAVIGATOR
# ------------------------------------------------

def navigate(query, top_k=3):

    query_embedding = model.encode(query, convert_to_tensor=True)

    scores = util.cos_sim(query_embedding, summary_embeddings)[0]

    scores = scores.tolist()

    # Financial keyword boost
    financial_keywords = [
        "revenue",
        "income",
        "profit",
        "ebitda",
        "financial",
        "statement",
        "balance sheet",
        "cash flow"
    ]

    boosted_scores = []

    for i, score in enumerate(scores):

        text = summary_texts[i].lower()

        boost = 0

        for keyword in financial_keywords:
            if keyword in text:
                boost += 0.15

        boosted_scores.append(score + boost)

    ranked = sorted(
        range(len(boosted_scores)),
        key=lambda i: boosted_scores[i],
        reverse=True
    )

    pages_found = []

    for idx in ranked[:top_k]:
        pages_found.append(page_lookup[idx])

    return pages_found


# ------------------------------------------------
# CONTEXT EXTRACTION
# ------------------------------------------------

def retrieve_pages(page_indices):

    results = []

    for page in pages:

        if page["Page_Index"] in page_indices:

            results.append(page["Content_MD"])

    return "\n\n".join(results)


# ------------------------------------------------
# SIMPLE QA
# ------------------------------------------------

def answer_query(query):

    page_indices = navigate(query)

    print("\nNavigator selected pages:", page_indices)

    context = retrieve_pages(page_indices)

    print("\nRetrieved Context Preview:\n")
    print(context[:1000])

    return context


# ------------------------------------------------
# DEMO LOOP
# ------------------------------------------------

if __name__ == "__main__":

    while True:

        query = input("\nAsk a question about the filing: ")

        if query.lower() == "exit":
            break

        answer_query(query)
