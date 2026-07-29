

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
# History-aware Retrieval Query
# ====================================================
def extract_keywords(query: str) -> str:
    """
    يستخرج الكلمات المفتاحية الجوهرية من السؤال (زي أسماء الجرائم أو المصطلحات القانونية)
    عشان تتضاف لاستعلام البحث وتحسن دقة الـ retrieval مع الأسئلة الطويلة.
    """

    messages = [
        {
            "role": "system",
            "content": """
استخرج فقط الكلمة أو الكلمات المفتاحية القانونية الجوهرية من السؤال التالي
(مثل اسم الجريمة أو المصطلح القانوني المحدد)، من غير أي كلمات سؤال زي "ما هي" أو "في القانون".

أخرج الكلمات فقط مفصولة بمسافة، بدون أي شرح.

مثال:
السؤال: ما هي عقوبة السرقة في القانون المصري؟
الإخراج: سرقة عقوبة

السؤال: ما حكم الطلاق للضرر؟
الإخراج: طلاق ضرر
"""
        },
        {"role": "user", "content": query}
    ]

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0,
            max_tokens=30
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return ""

def build_retrieval_query(query: str, chat_history: list) -> str:
    """
    يستخدم الـ LLM لتحديد هل السؤال الحالي يعتمد على الـ history
    أم هو سؤال جديد، ثم يعيد صياغة السؤال المناسب للـ Retrieval.
    """

    if not chat_history:
        return query

    # آخر 6 رسائل فقط لتقليل الـ tokens
    history = chat_history[-6:]

    messages = [
        {
            "role": "system",
            "content": """
أنت مسئول فقط عن إعادة كتابة السؤال لاستخدامه في البحث داخل قاعدة بيانات قانونية.

القواعد:

1- إذا كان السؤال يعتمد على المحادثة السابقة:
    - أعد كتابة السؤال ليصبح كاملاً ومستقلاً.
    - أضف المعلومات الناقصة من الـ history.
    - لا تضف أي معلومات غير موجودة.

2- إذا كان السؤال جديد ولا يعتمد على الـ history:
    - أعد السؤال كما هو بدون أي تعديل.

3- لا تجب على السؤال.
4- لا تشرح.
5- لا تضف أي كلام آخر.
6- أخرج السؤال فقط.
"""
        }
    ]

    messages.extend(history)

    messages.append({
        "role": "user",
        "content": query
    })

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0,
            max_tokens=150
        )

        retrieval_query = response.choices[0].message.content.strip()

        print("\n========== Retrieval Query ==========")
        print(retrieval_query)
        print("=====================================\n")

        return retrieval_query

    except Exception:
        return query

# ====================================================
# Retrieval
# ====================================================
def hybrid_retrieve(query, top_k=5, bm25_query=None):
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

    bm25_search_text = bm25_query if bm25_query else query   # ← الإضافة الوحيدة
    scores = bm25.get_scores(tokenize(bm25_search_text))
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
- لو فى غلط املاء صححوا علشان قواعد القانون لازم تكون واضحه 

طريقة الإجابة:
- حدد نوع الموضوع (جنائي / مدني / عمل / دستوري)
- استخرج الحكم القانوني بدقة مع رقم المادة إن وُجد
- اجعل الإجابة مختصرة وواضحة و بدون تكرار الجمل نهائى 
- التاكد من الاخطاء الاملائيه و تصحيحها
"""

def get_answer(query: str, chat_history: list) -> tuple[str, list]:
    """
    chat_history: list of {"role": "user"/"assistant", "content": "..."}
    returns: (answer, updated_history)
    """
    retrieval_query = build_retrieval_query(query, chat_history)

    keywords = extract_keywords(query)
    print(f"DEBUG - Keywords extracted: '{keywords}'")
    if keywords:
        retrieval_query = f"{retrieval_query} {keywords}"

    hits = hybrid_retrieve(retrieval_query, top_k=5, bm25_query=keywords if keywords else None)
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
        model="llama-3.3-70b-versatile",   # بدل llama-3.1-8b-instant
        messages=messages,
        temperature=0,
        max_tokens=600,
        frequency_penalty=0.3
    )

    answer = response.choices[0].message.content

    # ✅ نضيف السؤال والجواب للـ history
    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": answer})

    # ✅ نحتفظ بآخر 10 رسائل بس (5 exchanges) علشان ما نتجاوزش الـ context window
    if len(chat_history) > 10:
        chat_history = chat_history[-10:]

    return answer, chat_history, sources
