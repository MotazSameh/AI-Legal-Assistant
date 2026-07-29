import os
import re
import pdfplumber
import unicodedata

from pdf2image import convert_from_path
import pytesseract

# ====================================================
# Reverse Arabic Fix
# ====================================================

def fix_reversed_arabic(text):
    """
    يصلح النص العربي إذا كان خارج من الـ PDF معكوس.
    """

    def fix_line(line):

        reversed_line = line[::-1]

        reversed_line = re.sub(
            r"\d+",
            lambda m: m.group(0)[::-1],
            reversed_line
        )

        return reversed_line

    return "\n".join(
        fix_line(line)
        for line in text.split("\n")
    )


def is_reversed(text):
    """
    يكتشف إذا كان النص العربي معكوساً.
    """

    sample = text[:500]

    reversed_words = [

        "دقع",
        "جذومن",
        "يرتشم",
        "عئاب",
        "رجؤم",
        "رجاتسم",
        "فظوم",
        "لماع",
        "نوناق",
        "دقعلا",   
        "دنبلا",  
        "عيقوتلا"

    ]

    score = 0

    for word in reversed_words:

        if word in sample:
            score += 1

    return score >= 2


# ====================================================
# Font Corruption Detection (جديد)
# ====================================================

def is_font_corrupted(text):
    """
    يكتشف عطب الخط اللي بيحول "ال" لـ "اث" أو مشابه،
    بيحصل مع بعض ملفات الـ PDF المُصدَّرة بخط معطوب الـ CMap.
    """

    corrupted_patterns = [
        "اثبند", "اثطرف", "اثمشتري", "اثبائع",
        "اثعقار", "اثقاهرة", "اثجيزة", "اثدقي"
    ]

    if any(p in text for p in corrupted_patterns):
        return True

    # كاشف جديد: نسبة عالية من رموز يونيكود غريبة
    # (برّه نطاق العربي 0600-06FF و 0750-077F، اللاتيني، الأرقام، وعلامات الترقيم الشائعة)
    weird_chars = re.findall(
        r'[^\u0600-\u06FF\u0750-\u077F0-9A-Za-z\s.,،():\-]',
        text
    )

    if len(text) > 0 and (len(weird_chars) / len(text)) > 0.15:
        return True

    return False


# ====================================================
# OCR Fallback (جديد)
# ====================================================

def ocr_extract_page(pdf_path, page_num):
    """
    يحول صفحة واحدة من الـ PDF لصورة ويقرأها بـ OCR
    (يُستخدم فقط لو النص المستخرج مباشرة طلع معطوب)
    """

    try:
        images = convert_from_path(
            pdf_path,
            first_page=page_num + 1,
            last_page=page_num + 1
        )

        if not images:
            return ""

        return pytesseract.image_to_string(images[0], lang="ara")

    except Exception as e:
        print(f"OCR Error on page {page_num + 1}: {e}")
        return ""


# ====================================================
# Cleaning
# ====================================================

def clean_contract_text(text):

    text = unicodedata.normalize("NFKC", text)

    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"[ًٌٍَُِّْـ]", "", text)
    text = re.sub(r"[()（）]", " ", text)
    text = re.sub(
        r"[^\u0600-\u06FF0-9A-Za-z\s.,،():\-]",
        " ",
        text
    )

    text = " ".join(text.split())

    return text


# ====================================================
# Extract PDF
# ====================================================

def extract_contract_pdf(pdf_path):

    pages = []

    with pdfplumber.open(pdf_path) as pdf:

        for page_num, page in enumerate(pdf.pages):

            text = page.extract_text()

            if not text:
                continue

            if is_reversed(text):
                text = fix_reversed_arabic(text)

            # ← الإضافة الجديدة: لو النص لسه معطوب، نلجأ لـ OCR
            if is_font_corrupted(text):
                print(f"Font corruption detected on page {page_num + 1}, falling back to OCR")
                ocr_text = ocr_extract_page(pdf_path, page_num)
                if ocr_text.strip():
                    text = ocr_text

            text = clean_contract_text(text)

            pages.append({

                "page": page_num + 1,

                "text": text

            })

    return pages


# ====================================================
# Split Into Sections
# ====================================================

def extract_sections(text):

    text = re.sub(
        r"\(\s*(البند[^)]*)\)",
        r"\1",
        text
    )

    pattern = re.compile(
        r"(?="
        r"\bالبند\b|"
        r"\b\d+\s*[\.\-]\s*|"
        r"\bاولا\b|"
        r"\bثانيا\b|"
        r"\bثالثا\b|"
        r"\bرابعا\b|"
        r"\bخامسا\b|"
        r"\bسادسا\b|"
        r"\bسابعا\b|"
        r"\bثامنا\b|"
        r"\bتاسعا\b|"
        r"\bعاشرا\b"
        r")"
    )

    parts = pattern.split(text)

    sections = []

    for idx, part in enumerate(parts):

        part = part.strip()

        if len(part) < 20:
            continue

        if idx == 0:

            title = "المقدمة"

        else:

            match = re.match(
                r"(البند\s+[^\n]{0,40})",
                part
            )

            if not match:

                match = re.match(
                    r"(\d+\s*[\.\-])",
                    part
                )

            if not match:

                match = re.match(
                    r"(اولا|ثانيا|ثالثا|رابعا|خامسا|سادسا|سابعا|ثامنا|تاسعا|عاشرا)",
                    part
                )

            if match:

                title = match.group(1).strip()

            else:

                title = f"Section {idx}"

        sections.append({

            "id": idx,

            "title": title,

            "content": part,

            "word_count": len(part.split()),

            "char_count": len(part)

        })

    return sections


# ====================================================
# Parse PDF
# ====================================================

def parse_pdf(pdf_path):

    file_name = os.path.basename(pdf_path)

    pages = extract_contract_pdf(pdf_path)

    if not pages:
        return None

    full_text = "\n".join(

        page["text"]

        for page in pages

    )

    if is_reversed(full_text):

        full_text = fix_reversed_arabic(full_text)

        full_text = clean_contract_text(full_text)

    sections = extract_sections(full_text)

    return {

        "type": "pdf",

        "source": file_name,

        "pages": len(pages),

        "text": full_text,

        "sections": sections

    }


# ====================================================
# Parse Text
# ====================================================

def parse_text(text):

    if is_reversed(text):
        text = fix_reversed_arabic(text)

    cleaned = clean_contract_text(text)

    sections = extract_sections(cleaned)

    return {

        "type": "text",

        "source": None,

        "pages": 1,

        "text": cleaned,

        "sections": sections

    }


# ====================================================
# Main Parser
# ====================================================

def parse_document(document):

    if os.path.isfile(document):

        if document.lower().endswith(".pdf"):

            return parse_pdf(document)

        raise ValueError("Unsupported file type.")

    return parse_text(document)


# ====================================================
# Test
# ====================================================

if __name__ == "__main__":

    pdf = r"E:\3th_2\NLP\legal_ai_assistant\data\صيغة عقد إيجار شقة سكنية.pdf"

    result = parse_document(pdf)

    print("\n========== Document ==========\n")

    print("Type   :", result["type"])
    print("Pages  :", result["pages"])
    print("Source :", result["source"])

    print("\n========== Sections ==========\n")

    for section in result["sections"]:

        print("-" * 90)

        print(f"Section #{section['id']}")

        print("Title :", section["title"])

        print("Words :", section["word_count"])

        print()

        print(section["content"][:700])

        print()