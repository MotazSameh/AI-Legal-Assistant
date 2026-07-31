"""
سكريبت اختبار سريع: بيشغّل contract_parser.py على صورة (أو صور) العقد
وبيطبعلك نص section "بيانات الأطراف" زي ما طلع بالظبط بعد extract_sections()،
عشان نشوف هل رقم البطاقة القومية للطرف الأول موجود في النص المستخرج
ولا اتقطع/ضاع قبل ما يوصل للموديل.

قبل ما تشغّله:
عدّل السطر ده حسب الهيكل الفعلي لمشروعك (زي ما عملنا قبل كده):
"""

from backend.contract_parser import parse_document

# لو العقد صورة واحدة: حط مسارها هنا كـ string
# لو العقد صور متعددة (صفحة لكل صورة): حطهم كـ list بنفس ترتيب الصفحات
IMAGE_PATH = r"E:\3th_2\NLP\legal_ai_assistant\Photo\1.png"

# لو عندك أكتر من صفحة، استخدم ده بدل السطر اللي فوق:
# IMAGE_PATH = [
#     r"path/to/page1.jpg",
#     r"path/to/page2.jpg",
# ]


def find_matching_sections(sections, keywords):
    """
    بيدوّر في كل sections عن أي section عنوانه أو أول 100 حرف منه
    فيهم واحدة من الكلمات المفتاحية (زي "بيانات الأطراف"، "الطرف الأول").
    بنستخدم بحث نصي بسيط بدل ما نعتمد على title بس، لأن الـ title
    أحيانًا بييجي "Section N" لو الـ regex متطابقش مع صيغة العنوان الفعلية.
    """

    matches = []

    for s in sections:
        haystack = (s["title"] + " " + s["content"][:100])
        if any(k in haystack for k in keywords):
            matches.append(s)

    return matches


def main():

    result = parse_document(IMAGE_PATH)

    if result is None:
        print("فشل استخراج العقد — تأكد من مسار الصورة/الصور.")
        return

    print(f"Type: {result['type']} | Pages: {result['pages']} | Source: {result['source']}\n")

    sections = result["sections"]

    print(f"إجمالي عدد الـ sections المستخرجة: {len(sections)}\n")

    import re
    all_ids = re.findall(r"\d{10,14}", result["text"])
    print(f">>> كل الأرقام الطويلة (10-14 خانة) في النص الكامل المستخرج: {all_ids}\n")

    keywords = ["بيانات الاطراف", "بيانات الأطراف", "الطرف الاول", "الطرف الأول", "المؤجر"]

    matches = find_matching_sections(sections, keywords)

    if not matches:
        print("محدش لقى section يطابق كلمات بيانات الأطراف. هنطبع كل الـ sections عشان تشوف بنفسك:\n")
        matches = sections

    for s in matches:

        print("=" * 90)
        print(f"Section #{s['id']} — {s['title']}")
        print("-" * 90)
        print(s["content"])
        print()

        # فحص سريع: هل فيه أي رقم متتابع 14 خانة (شكل الرقم القومي المصري)؟
        import re
        ids_found = re.findall(r"\d{10,14}", s["content"])
        print(f">>> أرقام طويلة (10-14 خانة) لقيتها في النص ده: {ids_found}")
        print()


if __name__ == "__main__":
    main()