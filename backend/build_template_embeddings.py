import os
import json
import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # backend/
PROJECT_ROOT = os.path.dirname(BASE_DIR)                        # LEGAL_AI_ASSISTANT/
CHUNKS_DIR = os.path.join(PROJECT_ROOT, "data", "chunks")
CHROMA_PATH = os.path.join(PROJECT_ROOT, "chroma_db")

CONTRACT_TYPES = ["sale", "lease", "employment"]

model = SentenceTransformer("intfloat/multilingual-e5-base")
client = chromadb.PersistentClient(path=CHROMA_PATH)

for contract_type in CONTRACT_TYPES:

    chunks_path = os.path.join(CHUNKS_DIR, f"{contract_type}_chunks.json")

    if not os.path.exists(chunks_path):
        print(f"Skip {contract_type}: chunks file not found")
        continue

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    collection_name = f"contract_templates_{contract_type}"
    collection = client.get_or_create_collection(name=collection_name)

    if collection.count() > 0:
        print(f"{collection_name}: already has {collection.count()} chunks, skipping")
        continue

    texts = [f"passage: {c['text']}" for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True).tolist()

    metadatas = [{
        "template": c["template"],
        "section_id": c["section_id"],
        "title": c["title"]
    } for c in chunks]

    collection.add(
        documents=[c["text"] for c in chunks],
        embeddings=embeddings,
        metadatas=metadatas,
        ids=[c["id"] for c in chunks]
    )

    print(f"{collection_name}: inserted {len(chunks)} chunks")

print("\nDone building all template collections")