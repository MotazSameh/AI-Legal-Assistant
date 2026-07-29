import os
import json

# ==========================================================
# Paths
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ده يوصل للـ root الصح لو الملف جوه backend/
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

CONTRACTS_DIR = os.path.join(
    DATA_DIR,
    "contracts"
)

CHUNKS_DIR = os.path.join(
    DATA_DIR,
    "chunks"
)

LAWS_DIR = os.path.join(
    DATA_DIR,
    "laws"
)

LAW_CHUNKS = os.path.join(
    LAWS_DIR,
    "chunks.json"
)

print(CHROMA_PATH)
# ==========================================================
# Schema
# ==========================================================

def load_schema(contract_type):

    path = os.path.join(
        CONTRACTS_DIR,
        contract_type,
        f"{contract_type}.json"
    )

    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ==========================================================
# Template Chunks
# ==========================================================

def load_template_chunks(contract_type):

    path = os.path.join(
        CHUNKS_DIR,
        f"{contract_type}_chunks.json"
    )

    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ==========================================================
# Law Chunks
# ==========================================================

def load_law_chunks():

    print("Path:", LAW_CHUNKS)
    print("Exists:", os.path.exists(LAW_CHUNKS))

    if not os.path.exists(LAW_CHUNKS):
        return []

    with open(LAW_CHUNKS, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("Type:", type(data))
    print("Length:", len(data))

    if len(data):
        print(data[0])

    return data


# ==========================================================
# Retrieve Resources
# ==========================================================
import os
import json
import re
import chromadb
from sentence_transformers import SentenceTransformer

# ... (الكود القديم: load_schema, load_template_chunks, load_law_chunks)

# ==========================================================
# Setup
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

_model = None
_client = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("intfloat/multilingual-e5-base")
    return _model

def get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _client


def tokenize(text):
    return re.findall(r"[\w\u0600-\u06FF]+", text.lower())


# ==========================================================
# BM25 Index Cache (لكل collection على حدة)
# ==========================================================

_bm25_cache = {}   # { collection_name: {"bm25": ..., "texts": [...], "metas": [...]} }

def get_bm25_index(collection_name: str):
    """
    بيبني (أو يرجع من الكاش) BM25 index لأي collection،
    بنفس الطريقة اللي في chatbot.py بالظبط.
    """
    from rank_bm25 import BM25Okapi

    if collection_name in _bm25_cache:
        return _bm25_cache[collection_name]

    try:
        collection = get_client().get_collection(name=collection_name)
    except Exception:
        return None

    data = collection.get(include=["documents", "metadatas"])
    texts = data["documents"]
    metas = data["metadatas"]

    if not texts:
        return None

    bm25 = BM25Okapi([tokenize(t) for t in texts])

    _bm25_cache[collection_name] = {
        "bm25": bm25,
        "texts": texts,
        "metas": metas,
        "collection": collection
    }

    return _bm25_cache[collection_name]


# ==========================================================
# Generic Hybrid Retrieve (نفس منطق chatbot.py بالظبط)
# ==========================================================

def hybrid_retrieve(collection_name: str, query: str, top_k: int = 5):
    """
    نفس hybrid_retrieve بتاع chatbot.py (dense + BM25 + RRF)
    لكن شغالة على أي collection بالاسم بدل ما تكون مربوطة بـ legal_docs بس.
    """

    index = get_bm25_index(collection_name)

    if index is None:
        return []

    collection = index["collection"]
    bm25 = index["bm25"]
    all_texts = index["texts"]
    all_metas = index["metas"]

    query_emb = get_model().encode(
        "query: " + query, normalize_embeddings=True
    ).tolist()

    dense_results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    hits = {}
    for i, (doc, meta, dist) in enumerate(zip(
        dense_results["documents"][0],
        dense_results["metadatas"][0],
        dense_results["distances"][0]
    )):
        key = doc[:80]
        hits[key] = {"text": doc, "meta": meta, "rrf": 1 / (60 + i + 1)}

    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    for rank, (idx, score) in enumerate(ranked[:top_k]):
        if score == 0:
            continue
        key = all_texts[idx][:80]
        if key in hits:
            hits[key]["rrf"] += 1 / (60 + rank + 1)
        else:
            hits[key] = {
                "text": all_texts[idx],
                "meta": all_metas[idx],
                "rrf": 1 / (60 + rank + 1)
            }

    return sorted(hits.values(), key=lambda x: x["rrf"], reverse=True)[:top_k]


# ==========================================================
# Build Query From Schema (لحظة الرفع)
# ==========================================================

def build_validation_query(schema: dict) -> str:
    titles = [c["title"] for c in schema.get("required_clauses", [])]
    return " ، ".join(titles)


# ==========================================================
# Unified Retrieval Function
# ==========================================================

def retrieve_contract_resources(contract_type: str, query: str = None, top_k: int = 5):
    """
    query=None  -> وقت الرفع، بنبني query من الـ schema
    query=str   -> وقت سؤال إضافي من المستخدم
    """

    schema = load_schema(contract_type)

    if query is None:
        query = build_validation_query(schema) if schema else contract_type

    templates_collection = f"contract_templates_{contract_type}"

    return {
        "contract_type": contract_type,
        "schema": schema,
        "templates": hybrid_retrieve(templates_collection, query, top_k),
        "laws": hybrid_retrieve("legal_docs", query, top_k)
    }