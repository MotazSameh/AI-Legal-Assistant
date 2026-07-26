import json

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load documents
with open(r"E:\3th_2\NLP\legal_ai_assistant\documents.json", "r", encoding="utf-8") as f:
    documents = json.load(f)


# Create splitter
text_splitter = RecursiveCharacterTextSplitter(
    separators=[
        "\n\n",
        "\n",
        "مادة",
        ".",
        " "
    ],
    chunk_size=500,
    chunk_overlap=50
)


all_chunks = []


for doc in documents:
    chunks = text_splitter.split_text(doc["text"])
    for chunk in chunks:
        if len(chunk.strip()) < 30:
                    continue
        chunk_data = {
            "text": chunk,
            "page": doc["page"],
            "source": doc["source"]
        }

        all_chunks.append(chunk_data)


print("Total Chunks:", len(all_chunks))


with open(r"E:\3th_2\NLP\legal_ai_assistant\chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=4)


print("\nChunks saved successfully")