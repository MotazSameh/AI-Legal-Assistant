import json
import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/multilingual-e5-base")
client = chromadb.PersistentClient(path=r"E:\3th_2\NLP\legal_ai_assistant\chroma_db")
collection = client.get_or_create_collection(name="legal_docs")

if collection.count() > 0:
    print("Data already exists, skipping insert")
    print("Total stored chunks:", collection.count())
    exit()

with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

print("Total chunks:", len(chunks))

BATCH_SIZE = 64

for start in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[start:start + BATCH_SIZE]
    texts = [f"passage: {c['text']}" for c in batch]
    embeddings = model.encode(texts, normalize_embeddings=True).tolist()

    metadatas = []
    for c in batch:
        metadatas.append({
            "page": c["page"],
            "source": c["source"],
            "article": c["article"] if c["article"] is not None else "",  # ← Chroma بيرفض None
            "part": c["part"],
            "total_parts": c["total_parts"]
        })

    collection.add(
        documents=[c["text"] for c in batch],
        embeddings=embeddings,
        metadatas=metadatas,
        ids=[str(start + i) for i in range(len(batch))]
    )
    print(f"Inserted {start + len(batch)}/{len(chunks)}")

print("\nAll data inserted successfully")