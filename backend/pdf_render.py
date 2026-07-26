"""
from pypdf import PdfReader
import unicodedata
import os


DATA_PATH = "data"


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)

    full_text = ""

    for page_number, page in enumerate(reader.pages):
        text = page.extract_text()

        if text:
            full_text += f"\n--- Page {page_number + 1} ---\n"
            cleaned_text = clean_arabic_text(text)
            full_text += cleaned_text

    return full_text


def read_all_pdfs():
    for file_name in os.listdir(DATA_PATH):

        if file_name.endswith(".pdf"):

            pdf_path = os.path.join(DATA_PATH, file_name)

            print("=" * 50)
            print(f"Reading: {file_name}")
            print("=" * 50)

            text = extract_text_from_pdf(pdf_path)

            output_file = file_name.replace(".pdf", ".txt")

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(text)

            print(f"{output_file} saved successfully")   # أول 3000 حرف فقط
            print("\n\n")

def clean_arabic_text(text):

    text = unicodedata.normalize("NFKC", text)
    text = " ".join(text.split())

    return text


if __name__ == "__main__":
    read_all_pdfs()
"""



from pypdf import PdfReader
import pdfplumber
import os
import unicodedata
import json

DATA_PATH = r"E:\3th_2\NLP\legal_ai_assistant\data"


import re

import re
import unicodedata

def clean_arabic_text(text):
    # Unicode normalize
    text = unicodedata.normalize("NFKC", text)
    # توحيد الألف
    text = re.sub(r"[إأآا]", "ا", text)
    # توحيد الياء
    text = re.sub(r"ى", "ي", text)
    # إزالة التشكيل
    text = re.sub(r"[ًٌٍَُِّْـ]", "", text)
    # إصلاح "مادة1149"
    text = re.sub(r"مادة(\d+)", r"مادة \1", text)
    # إزالة الرموز الغريبة
    text = re.sub(r"[^\u0600-\u06FF0-9\s]", " ", text)
    #خلّي الأرقام الإنجليزية والنقط والفواصل
    text = re.sub(r"[^\u0600-\u06FF0-9A-Za-z\s.,،():-]", " ", text)
    # إزالة المسافات الزائدة
    text = " ".join(text.split())
    return text

"""
def extract_text_from_pdf(pdf_path, file_name):
    reader = PdfReader(pdf_path)
    documents = []
    for page_number, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            cleaned_text = clean_arabic_text(text)
            document = {
                "text": cleaned_text,
                "page": page_number + 1,
                "source": file_name
            }
            documents.append(document)
    return documents
"""

def extract_text_from_pdf(pdf_path, file_name):
    documents = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    cleaned = clean_arabic_text(text)
                    if len(cleaned) > 20:  # تجاهل صفحات فاضية
                        documents.append({
                            "text": cleaned,
                            "page": page_number + 1,
                            "source": file_name
                        })
    except Exception as e:
        print(f" Error reading {file_name}: {e}")
    return documents

def read_all_pdfs():
    all_documents = []
    for file_name in os.listdir(DATA_PATH):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(DATA_PATH, file_name)
            print(f"\nReading: {file_name}")
            documents = extract_text_from_pdf(pdf_path, file_name)
            all_documents.extend(documents)
    return all_documents


if __name__ == "__main__":

    documents = read_all_pdfs()

    print("\nTotal Pages:", len(documents))

    with open("documents.json", "w", encoding="utf-8") as f:
         json.dump(documents, f, ensure_ascii=False, indent=4)

    print("\nDocuments saved successfully")