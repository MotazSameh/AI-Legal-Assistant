import os
import re
import base64
import time
import unicodedata
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

import pdfplumber
from pdf2image import convert_from_path
import cv2
import numpy as np

from groq import Groq, RateLimitError, APIStatusError
from dotenv import load_dotenv

load_dotenv()

# ====================================================
# Groq Vision OCR Client
# ====================================================

GROQ_MODEL = "qwen/qwen3.6-27b"  # موديل الـ vision الوحيد المتاح حاليًا على Groq (27B, multimodal, يدعم OCR)

_groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

OCR_PROMPT = (
    "استخرج كل النص الموجود في هذه الصورة بالضبط كما هو مكتوب، بدون أي ترجمة أو تلخيص أو تعليق. "
    "حافظ على ترتيب الأسطر والفقرات كما تظهر في الصورة تمامًا. "
    "لا تستخرج أرقام الصفحات (page numbers) مهما كان شكلها، ولا أي ترويسة (header) أو تذييل (footer) "
    "متكرر شكليًا وغير جزء من متن العقد نفسه (مثل رقم صفحة منفرد، أو اسم ملف، أو خط فاصل زخرفي). "
    "استخرج فقط النص الفعلي لبنود العقد ومحتواه. "
    "أرجع النص المستخرج فقط، بدون أي مقدمة أو شرح إضافي من عندك."
)


def encode_pil_image(image, max_dimension=1600):
    """
    يحول صورة PIL (من pdf2image) إلى base64 string.
    بيصغّر الصورة لو أبعادها أكبر من max_dimension، عشان يقلل عدد
    التوكنز اللي هيستهلكها الموديل (نفس منطق preprocess_image_light).
    """

    w, h = image.size
    if max(w, h) > max_dimension:
        scale = max_dimension / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)))

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def encode_image_file(image_path):
    """
    يحول ملف صورة لـ base64 مباشرة زي ما هو، من غير أي preprocessing
    (تصغير/تكبير/denoise). ده اللي طلع أدق وأسرع في التجربة الفعلية،
    فبقى هو الافتراضي لمسار الصور.
    """

    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def encode_cv_image(cv_img):
    """يحول صورة OpenCV (numpy array) إلى base64 string. (متسابة هنا لو احتجتها لاحقًا)"""

    success, buffer = cv2.imencode(".png", cv_img)

    if not success:
        return None

    return base64.b64encode(buffer).decode("utf-8")


def groq_ocr_from_base64(base64_str, max_retries=3):
    """
    يبعت الصورة (base64) لموديل Groq Vision ويرجع النص المستخرج.
    فيه إعادة محاولة (retry) مع انتظار متزايد لو حصل rate limit،
    بدل ما يرجع نص فاضي بهدوء ويضيّع الصفحة دي من النتيجة النهائية.
    """

    for attempt in range(max_retries):

        try:
            completion = _groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": OCR_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_str}"
                                },
                            },
                        ],
                    }
                ],
                temperature=0,  # 0 عشان نضمن استخراج حرفي، مش إبداعي
                max_completion_tokens=4096,
                reasoning_effort="none",  # وضع non-thinking: رد مباشر من غير خطوات تفكير داخلية غير ضرورية لمهمة استخراج نص بسيطة
            )

            return completion.choices[0].message.content or ""

        except RateLimitError as e:
            wait_time = 2 ** attempt  # 1s, 2s, 4s
            print(f"⚠️ Rate limit من Groq (محاولة {attempt + 1}/{max_retries})، هستنى {wait_time} ثانية...")
            time.sleep(wait_time)

        except APIStatusError as e:
            print(f"❌ Groq API Error (status {e.status_code}): {e.message}")
            return ""

        except Exception as e:
            print(f"❌ Groq OCR Error غير متوقع: {e}")
            return ""

    print("❌ فشلت كل المحاولات بسبب rate limit — الصفحة دي هترجع فاضية")
    return ""


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
        "نوناق"

    ]

    score = 0

    for word in reversed_words:

        if word in sample:
            score += 1

    return score >= 2


# ====================================================
# Font Corruption Detection
# ====================================================

def is_font_corrupted(text):
    """
    يكتشف عطب الخط بنوعين:
    1. النمط المعروف: "ال" بتتحول لـ "اث" أو مشابه
    2. نمط عام: ظهور كثيف لرموز يونيكود خارج نطاق العربي/اللاتيني/الأرقام
       (زي ما يحصل مع بعض ملفات الـ PDF ذات الـ CMap المعطوب بشكل أعمق)
    """

    corrupted_patterns = [
        "اثبند", "اثطرف", "اثمشتري", "اثبائع",
        "اثعقار", "اثقاهرة", "اثجيزة", "اثدقي"
    ]

    if any(p in text for p in corrupted_patterns):
        return True

    if not text.strip():
        return False

    weird_chars = re.findall(
        r"[^\u0600-\u06FF\u0750-\u077F0-9A-Za-z\s.,،():\-]",
        text
    )

    ratio = len(weird_chars) / max(len(text), 1)

    return ratio > 0.15


# ====================================================
# OCR Fallback (PDF pages) — via Groq Vision
# ====================================================

def ocr_extract_page(pdf_path, page_num):
    """
    يحول صفحة واحدة من الـ PDF لصورة ويقرأها بـ Groq Vision OCR
    (يُستخدم فقط لو النص المستخرج مباشرة طلع معطوب)
    """

    try:
        images = convert_from_path(
            pdf_path,
            first_page=page_num + 1,
            last_page=page_num + 1,
            dpi=200  # كافية لوضوح النص العربي، من غير ما تنتج صورة ضخمة تستهلك توكنز زيادة
        )

        if not images:
            return ""

        b64_image = encode_pil_image(images[0])

        return groq_ocr_from_base64(b64_image)

    except Exception as e:
        print(f"OCR Error on page {page_num + 1}: {e}")
        return ""


# ====================================================
# Image Preprocessing
# ====================================================

def preprocess_image_light(image_path, max_dimension=1600):
    """
    تحسين خفيف مخصوص لموديلات الـ Vision (VLM) زي Groq/Qwen.

    مهم جدًا: عمدًا من غير أي تكبير (upscale) للصور الصغيرة، لأن موديلات
    Qwen بتحوّل الصورة لتوكنز حسب عدد البكسلات الفعلي (طول × عرض تقريبًا).
    تكبير صورة صغيرة معناه "توكنز" إضافية على بيانات متخيّلة (interpolated)
    مش تفاصيل حقيقية جديدة — يعني تكلفة ووقت زيادة من غير أي فايدة في الدقة.

    الوحيد اللي بنعمله هنا: تصغير (downscale) لو الصورة كبيرة أوي، عشان
    نتجنب استهلاك توكنز زيادة من غير داعي، ومن غير Otsu/Adaptive Threshold
    (binarization) لأن الـ VLM بيفهم الصورة بصريًا مش بمطابقة بكسلات.
    """

    img = cv2.imread(image_path)

    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    if max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    denoised = cv2.fastNlMeansDenoising(gray, h=7)

    return denoised


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
# Page Artifact Cleanup
# ====================================================

def strip_page_number_artifacts(text):
    """
    يشيل الأسطر اللي هي مجرد رقم صفحة (أو رقم صفحة مزخرف زي "- 1 -"،
    "1 / 10"، "صفحة 1")، لأن الرقم ده لو سيبناه هيقع لوحده وسط جملة
    من العقد بمجرد ما نلزّق نص الصفحات ببعض بـ "\n".join(...) — وده
    اللي كان بيخلي judge_clause يشوف نص "مدة العقد" مقطوع برقم غريب
    وسطه ويحكم عليه إنه غامض.

    بيتطبق على كل صفحة لوحدها *قبل* اللزق، عشان نضمن إن الرقم مش هيلزق
    غلط جوه جملة من صفحة تانية.
    """

    if not text:
        return text

    cleaned_lines = []

    for line in text.split("\n"):

        stripped = line.strip()

        if not stripped:
            cleaned_lines.append(line)
            continue

        # سطر بيتكون بس من رقم قصير (رقم صفحة)، ممكن محاط بشرطات/نقط زخرفية
        if re.fullmatch(r"[-–—.\s]{0,5}\d{1,4}[-–—.\s]{0,5}", stripped):
            continue

        # صيغ صريحة زي "صفحة 3" أو "Page 3" أو "3 / 12"
        if re.fullmatch(
            r"(صفحة|page)\s*\d{1,4}(\s*(من|/|-)\s*\d{1,4})?",
            stripped,
            flags=re.IGNORECASE
        ):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


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

            text = strip_page_number_artifacts(text)

            if is_reversed(text):
                text = fix_reversed_arabic(text)

            # لو النص لسه معطوب بعد الإصلاح المعتاد، نلجأ لـ OCR (Groq Vision)
            if is_font_corrupted(text):
                print(f"Font corruption detected on page {page_num + 1}, falling back to Groq OCR")
                ocr_text = ocr_extract_page(pdf_path, page_num)
                if ocr_text.strip():
                    text = strip_page_number_artifacts(ocr_text)

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
        r"(?<!\d)\d{1,2}[\.\-]\s|"
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
                    r"(\d{1,2}\s*[\.\-])",
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
# Parse Single Image — via Groq Vision
# ====================================================

def parse_image(image_path):
    """
    يشغّل Groq Vision OCR مباشرة على صورة مرفوعة، بيبعتها زي ما هي
    من غير أي preprocessing (جرّبنا التصغير/denoise وطلع بيقلل الدقة
    مع VLM، فالصورة الخام أدق وأسرع).
    """

    b64_image = encode_image_file(image_path)

    raw_text = groq_ocr_from_base64(b64_image)

    if not raw_text.strip():
        return None

    raw_text = strip_page_number_artifacts(raw_text)

    if is_reversed(raw_text):
        raw_text = fix_reversed_arabic(raw_text)

    text = clean_contract_text(raw_text)
    sections = extract_sections(text)

    return {

        "type": "image",

        "source": os.path.basename(image_path),

        "pages": 1,

        "text": text,

        "sections": sections

    }


# ====================================================
# Parse Multiple Images (عقد بصفحات متعددة كصور منفصلة) — via Groq Vision
# ====================================================

def _ocr_single_image_for_batch(indexed_path):
    """
    دالة مساعدة تعالج صورة واحدة (لاستخدامها جوه ThreadPoolExecutor).
    بترجع (index, source_name, cleaned_text أو None).
    بتبعت الصورة زي ما هي من غير preprocessing (نفس منطق parse_image).
    """

    idx, image_path = indexed_path

    b64_image = encode_image_file(image_path)

    raw_text = groq_ocr_from_base64(b64_image)

    if not raw_text.strip():
        return idx, os.path.basename(image_path), None

    raw_text = strip_page_number_artifacts(raw_text)

    if is_reversed(raw_text):
        raw_text = fix_reversed_arabic(raw_text)

    cleaned = clean_contract_text(raw_text)

    return idx, os.path.basename(image_path), cleaned if cleaned.strip() else None


def parse_multiple_images(image_paths: list, max_workers: int = 1):
    """
    بتاخد قائمة مسارات صور (كل صورة = صفحة من نفس العقد)،
    تشغّل Groq Vision OCR على الصور، وتدمج النصوص بترتيب الصفحات الأصلي.

    max_workers الافتراضي بقى 1 (تسلسلي) مش 3، لأن اللي بيضربنا فعليًا
    هو حد الـ TPM (توكنز في الدقيقة) بتاع Free tier، مش عدد الطلبات بس.
    كل صورة بتاخد مئات-آلاف التوكنز، فحتى تزامن بسيط بيتخطى الحد بسرعة.
    لو رفعت حسابك لـ Developer tier (بطاقة ائتمان، من غير رسوم مضمونة)،
    ممكن ترجع max_workers لـ 3-5 براحة.
    """

    indexed_paths = list(enumerate(image_paths))

    results = [None] * len(image_paths)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        futures = {
            executor.submit(_ocr_single_image_for_batch, item): item[0]
            for item in indexed_paths
        }

        for future in as_completed(futures):

            idx, source_name, cleaned_text = future.result()

            results[idx] = (source_name, cleaned_text)

    all_page_texts = []
    sources = []
    failed_sources = []

    # بنمشي بالترتيب الأصلي (0, 1, 2, ...) عشان الصفحات متترصش عشوائي
    for source_name, cleaned_text in results:

        if cleaned_text:
            all_page_texts.append(cleaned_text)
            sources.append(source_name)
        else:
            failed_sources.append(source_name)

    if failed_sources:
        print(f"⚠️ الصور دي فشلت تمامًا ولم تُستخرج منها أي نص: {failed_sources}")

    if not all_page_texts:
        return None

    full_text = "\n".join(all_page_texts)

    sections = extract_sections(full_text)

    return {

        "type": "image_multi",

        "source": ", ".join(sources),

        "pages": len(all_page_texts),

        "text": full_text,

        "sections": sections

    }


# ====================================================
# Main Parser
# ====================================================

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")


def parse_document(document):
    """
    document ممكن يكون:
    - مسار ملف PDF (string)
    - مسار صورة واحدة (string)
    - قائمة مسارات صور (list) لعقد متعدد الصفحات
    - نص خام (string مش مسار ملف)
    """

    if isinstance(document, list):
        return parse_multiple_images(document)

    if os.path.isfile(document):

        lower_path = document.lower()

        if lower_path.endswith(".pdf"):
            return parse_pdf(document)

        if lower_path.endswith(IMAGE_EXTENSIONS):
            return parse_image(document)

        raise ValueError("Unsupported file type.")

    return parse_text(document)


# ====================================================
# Test
# ====================================================

if __name__ == "__main__":

    pdf = r"E:\3th_2\NLP\legal_ai_assistant\data\نموذج عقد بيع ابتدائي.pdf"

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