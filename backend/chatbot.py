"""
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq
from rank_bm25 import BM25Okapi
import re
import os

# ====================================================
# API
# ====================================================
client_groq = Groq(api_key="Key")

# ====================================================
# Models
# ====================================================
embedding_model = SentenceTransformer("intfloat/multilingual-e5-base")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ====================================================
# Vector DB
# ====================================================
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="legal_docs")

# ====================================================
# Cleaning
# ====================================================
def clean_text(text):
    text = re.sub(r"[^\u0600-\u06FF0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def tokenize(text):
    return re.findall(r"[\w\u0600-\u06FF]+", text.lower())

# ====================================================
# Build BM25
# ====================================================
all_data = collection.get(include=["documents", "metadatas"])
all_texts = [clean_text(t) for t in all_data["documents"]]
all_metas = all_data["metadatas"]

bm25 = BM25Okapi([tokenize(t) for t in all_texts])

# ====================================================
# Hybrid Retrieve
# ====================================================
def hybrid_retrieve(query, top_k=5):
    query_clean = clean_text(query)

    # Dense (E5 FIX)
    query_emb = embedding_model.encode(
        "query: " + query_clean,
        normalize_embeddings=True
    ).tolist()

    dense_results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    dense_hits = {}

    for i, (doc, meta, dist) in enumerate(zip(
        dense_results["documents"][0],
        dense_results["metadatas"][0],
        dense_results["distances"][0]
    )):
        key = doc[:80]
        dense_hits[key] = {
            "text": clean_text(doc),
            "meta": meta,
            "rrf": 1 / (60 + i + 1)
        }

    # BM25
    scores = bm25.get_scores(tokenize(query_clean))
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    for rank, (idx, score) in enumerate(ranked[:top_k]):
        if score == 0:
            continue

        key = all_texts[idx][:80]

        if key in dense_hits:
            dense_hits[key]["rrf"] += 1 / (60 + rank + 1)
        else:
            dense_hits[key] = {
                "text": all_texts[idx],
                "meta": all_metas[idx],
                "rrf": 1 / (60 + rank + 1)
            }

    final = sorted(dense_hits.values(), key=lambda x: x["rrf"], reverse=True)
    return final[:top_k]

# ====================================================
# Reranker
# ====================================================
def rerank(query, hits):
    pairs = [[query, h["text"]] for h in hits]
    scores = reranker.predict(pairs)

    for i in range(len(hits)):
        hits[i]["score"] = scores[i]

    return sorted(hits, key=lambda x: x["score"], reverse=True)

# ====================================================
# Prompt
# ====================================================
"""
SYSTEM_PROMPT = """أنت مساعد قانوني مصري خبير في جميع فروع القانون.

قواعد صارمة:
- أجب فقط من النص المقدم
- لا تخترع أي معلومة
- إذا الإجابة غير موجودة: "غير مذكور في النص المتاح"
- إذا السؤال غير واضح، اطلب توضيح

طريقة الإجابة:
- حدد نوع الموضوع (جنائي / مدني / عمل / دستوري إن أمكن)
- استخرج الحكم القانوني بدقة
- اذكر العقوبة أو الحكم إن وُجد
- اذكر رقم المادة إن وُجد
- اجعل الإجابة مختصرة وواضحة
"""
"""
# ====================================================
# Main Loop
# ====================================================
print("اكتب 'خروج' للإنهاء\n")

while True:


    query = input("اكتب سؤالك: ").strip()

    if query == "خروج":
        break

    # Retrieve
    hits = hybrid_retrieve(query, top_k=5)

    # Rerank
    hits = rerank(query, hits)[:3]

    # Build Context
    context = ""
    for hit in hits:
        context += f"\n[{hit['meta'].get('source','؟')} - صفحة {hit['meta'].get('page','؟')}]\n{hit['text']}\n"

    # Generate Answer
    response = client_groq.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"النص القانوني:\n{context}\n\nالسؤال: {query}"}
        ],
        temperature=0
    )

    print("\n=== Answer ===\n")
    print(response.choices[0].message.content)
    print()

"""


import os
import re
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from groq import Groq
from dotenv import load_dotenv  

load_dotenv()


# ====================================================
# Init
# ====================================================
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))  
embedding_model = SentenceTransformer("intfloat/multilingual-e5-base")

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="legal_docs")

# ====================================================
# BM25 Setup
# ====================================================
def tokenize(text):
    return re.findall(r"[\w\u0600-\u06FF]+", text.lower())

all_data = collection.get(include=["documents", "metadatas"])
all_texts = all_data["documents"]
all_metas = all_data["metadatas"]
bm25 = BM25Okapi([tokenize(t) for t in all_texts])

# ====================================================
# Retrieval
# ====================================================
def hybrid_retrieve(query, top_k=5):
    query_emb = embedding_model.encode(
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
            hits[key] = {"text": all_texts[idx], "meta": all_metas[idx],
                         "rrf": 1 / (60 + rank + 1)}

    return sorted(hits.values(), key=lambda x: x["rrf"], reverse=True)[:top_k]

# ====================================================
# Chat History
# ====================================================
SYSTEM_PROMPT = """أنت مساعد قانوني مصري خبير في جميع فروع القانون.

قواعد صارمة:
- أجب فقط من النص المقدم
- لا تخترع أي معلومة
- إذا الإجابة غير موجودة: قل "غير مذكور في النص المتاح"
- إذا السؤال غير واضح، اطلب توضيح

طريقة الإجابة:
- حدد نوع الموضوع (جنائي / مدني / عمل / دستوري)
- استخرج الحكم القانوني بدقة مع رقم المادة إن وُجد
- اجعل الإجابة مختصرة وواضحة
"""

def get_answer(query: str, chat_history: list) -> tuple[str, list]:
    """
    chat_history: list of {"role": "user"/"assistant", "content": "..."}
    returns: (answer, updated_history)
    """
    hits = hybrid_retrieve(query, top_k=5)
    sources = []
    print("\n=== Retrieved Chunks ===")

    for i, hit in enumerate(hits):
            print(f"\n[{i+1}] Source: {hit['meta'].get('source')} | Page: {hit['meta'].get('page')}")
            print(f"Text: {hit['text'][:200]}")
    print("========================\n")
    context = ""
    for hit in hits[:3]:
            context += f"\n[{hit['meta'].get('source','؟')} - صفحة {hit['meta'].get('page','؟')}]\n{hit['text']}\n"
            sources.append({                              # ✅
                "source": hit['meta'].get('source', ''),
                "page": hit['meta'].get('page', 1)
            })
    # ✅ بنبني الـ messages كاملة مع الـ history
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += chat_history  # السياق السابق
    messages.append({
        "role": "user",
        "content": f"النص القانوني المرجعي:\n{context}\n\nالسؤال: {query}"
    })

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0,
        max_tokens=1024
    )

    answer = response.choices[0].message.content

    # ✅ نضيف السؤال والجواب للـ history
    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": answer})

    # ✅ نحتفظ بآخر 10 رسائل بس (5 exchanges) علشان ما نتجاوزش الـ context window
    if len(chat_history) > 10:
        chat_history = chat_history[-10:]

    return answer, chat_history, sources
