import json
import chromadb
from sentence_transformers import SentenceTransformer

# تحميل الموديل
model = SentenceTransformer("intfloat/multilingual-e5-base")

#  Persistent client (المهم)
client = chromadb.PersistentClient(path=r"E:\3th_2\NLP\legal_ai_assistant\chroma_db")

# collection
collection = client.get_or_create_collection(name="legal_docs")

# check
if collection.count() > 0:
    print("Data already exists, skipping insert ")
    print("Total stored chunks:", collection.count())
    exit()

# تحميل chunks
with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

print("Total chunks:", len(chunks))
"""
# insert
for i, chunk in enumerate(chunks):

    text = chunk["text"]

    embedding = model.encode(text).tolist()

    collection.add(
        documents=[text],
        embeddings=[embedding],
        metadatas=[{
            "page": chunk["page"],
            "source": chunk["source"]
        }],
        ids=[str(i)]
    )

    if i % 100 == 0:
        print(f"Inserted {i} chunks")
"""
BATCH_SIZE = 64  

for start in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[start:start + BATCH_SIZE]
    texts = [f"passage: {c['text']}" for c in batch]  # ✅ E5 prefix مهم
    embeddings = model.encode(texts, normalize_embeddings=True).tolist()

    collection.add(
        documents=[c["text"] for c in batch],
        embeddings=embeddings,
        metadatas=[{"page": c["page"], "source": c["source"]} for c in batch],
        ids=[str(start + i) for i in range(len(batch))]
    )
    print(f"Inserted {start + len(batch)}/{len(chunks)}")


print("\nAll data inserted successfully ")