import chromadb
from sentence_transformers import SentenceTransformer

# تحميل الموديل
model = SentenceTransformer("intfloat/multilingual-e5-base")

client = chromadb.PersistentClient(path="./chroma_db")

# تحميل الـ collection
collection = client.get_collection(name="legal_docs")

# سؤال المستخدم
query = "ما هي عقوبة القتل؟"

# تحويله لـ embedding
query_embedding = model.encode(query).tolist()

# search
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

print("\nResults:\n")

for i in range(len(results["documents"][0])):
    print(f"Result {i+1}:")
    print("Text:", results["documents"][0][i])
    print("Metadata:", results["metadatas"][0][i])
    print("-" * 50)