import os
import json

from contract_parser import parse_document

# ==========================================================
# Paths
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEMPLATE_DIR = os.path.join(
    BASE_DIR,
    "data",
    "contracts"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "chunks"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# Extract Section Title
# ==========================================================

def get_section_title(text):

    first = text.split("\n")[0].strip()

    if first.startswith("البند"):

        return first

    return "المقدمة"


# ==========================================================
# Build Chunks
# ==========================================================

def build_chunks(contract_type):

    folder = os.path.join(
        TEMPLATE_DIR,
        contract_type
    )

    chunks = []

    chunk_id = 1

    for file in os.listdir(folder):

        if not file.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(folder, file)

        print(f"Parsing {file}")

        document = parse_document(pdf_path)

        if document is None:
            continue

        for section in document["sections"]:

            chunk = {

                "id": f"{contract_type}_{chunk_id:04d}",

                "contract_type": contract_type,

                "template": file,

                "page": None,

                "section_id": section["id"],

                "title": get_section_title(
                    section["content"]
                ),

                "text": section["content"],

                "embedding": None

            }

            chunks.append(chunk)

            chunk_id += 1

    output = os.path.join(
        OUTPUT_DIR,
        f"{contract_type}_chunks.json"
    )

    with open(
        output,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            chunks,
            f,
            ensure_ascii=False,
            indent=4
        )

    print(f"\nSaved {len(chunks)} chunks -> {output}")

    return chunks


# ==========================================================
# Build All
# ==========================================================

def build_all():

    contract_types = [

        "sale",
        "lease",
        "employment"

    ]

    for contract_type in contract_types:

        print("=" * 70)

        print(contract_type.upper())

        print("=" * 70)

        build_chunks(contract_type)


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    build_all()