import json
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

with open(r"E:\3th_2\NLP\legal_ai_assistant\documents.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

# سبلتر تانوي بس لو المادة نفسها طويلة أوي (مفيش "مادة" في الـ separators هنا نهائيًا)
sub_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ".", " "],
    chunk_size=500,
    chunk_overlap=50
)

# بيمسك "مادة" متبوعة برقم، ويحتفظ بالـ delimiter نفسه في نتيجة الـ split
ARTICLE_SPLIT_RE = re.compile(r"(مادة\s*\d+)")

def split_by_article(text: str):
    """
    يرجع list من dicts: {"article": "558" أو None, "text": "..."}
    أي نص قبل أول 'مادة' في الصفحة بيتحط article=None (تمهيد/عنوان الباب مثلاً)
    """
    parts = ARTICLE_SPLIT_RE.split(text)
    # parts هتبقى: [قبل_أول_مادة, "مادة 558", نص_558, "مادة 559", نص_559, ...]

    results = []

    # أي نص قبل أول "مادة" (لو موجود ومفيدش فاضي)
    if parts[0].strip():
        results.append({"article": None, "text": parts[0].strip()})

    # من بعد كده هنمشي بخطوة 2: (marker, text)
    i = 1
    while i < len(parts):
        marker = parts[i]                      # "مادة 558"
        article_num = re.search(r"\d+", marker).group()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        full_text = f"{marker} {body}".strip()
        if full_text:
            results.append({"article": article_num, "text": full_text})
        i += 2

    return results


all_chunks = []

for doc in documents:
    articles = split_by_article(doc["text"])

    for art in articles:
        article_num = art["article"]
        article_text = art["text"]

        if len(article_text) <= 500:
            # المادة قصيرة، chunk واحد بس
            if len(article_text.strip()) < 30:
                continue
            all_chunks.append({
                "text": article_text,
                "page": doc["page"],
                "source": doc["source"],
                "article": article_num,   # موثوق 100% - مش تخمين
                "part": 1,
                "total_parts": 1
            })
        else:
            # المادة طويلة، قسّمها لسب-تشانكس لكن نفس رقم المادة يتورّث للكل
            sub_chunks = sub_splitter.split_text(article_text)
            sub_chunks = [c for c in sub_chunks if len(c.strip()) >= 30]
            for idx, sub in enumerate(sub_chunks, start=1):
                all_chunks.append({
                    "text": sub,
                    "page": doc["page"],
                    "source": doc["source"],
                    "article": article_num,
                    "part": idx,
                    "total_parts": len(sub_chunks)
                })

print("Total Chunks:", len(all_chunks))

# إحصائية سريعة تفيدك تتأكد إن الفصل نجح
no_article_count = sum(1 for c in all_chunks if c["article"] is None)
print(f"Chunks بدون رقم مادة (تمهيدات/عناوين): {no_article_count}")

with open(r"E:\3th_2\NLP\legal_ai_assistant\chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=4)

print("\nChunks saved successfully")