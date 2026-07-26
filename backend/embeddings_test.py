from sentence_transformers import SentenceTransformer

# تحميل الموديل
model = SentenceTransformer("all-MiniLM-L6-v2")

# مثال chunk (خده من chunks.json عندك)
text = "مادة 1149 اقتسم الذين للشرآء في حق من القسمة..."

# تحويل النص لـ embedding
embedding = model.encode(text)

print("Embedding length:", len(embedding))
print("\nFirst 10 values:\n")
print(embedding[:10])