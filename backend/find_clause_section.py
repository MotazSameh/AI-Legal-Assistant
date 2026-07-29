import os
import json
from groq import Groq
from dotenv import load_dotenv

from .contract_retriever import load_schema, hybrid_retrieve
load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ==========================================================
# Step 1: Match each clause to its section(s) in the contract
# ==========================================================

import re

def normalize_for_matching(text: str) -> str:
    """
    بيطبّع النص بنفس منطق clean_contract_text (الهمزات والألف)
    عشان نضمن تطابق search_terms مع النص المُنضّف بالظبط.
    """
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    return text

"""
    بيدور في sections العقد المرفوع (من contract_parser.py)
    عن أي section فيه أي search_term بتاع الـ clause.
    بيرجع النص المجمع لكل الـ sections اللي فيها تطابق، أو None لو مفيش.
"""


def find_clause_section(clause: dict, sections: list) -> str:

    matched_texts = []

    for section in sections:
        content = section["content"]
        normalized_content = normalize_for_matching(content)

        for term in clause.get("search_terms", []):
            normalized_term = normalize_for_matching(term)
            if normalized_term in normalized_content:
                matched_texts.append(content)
                break

    if not matched_texts:
        return None

    return "\n---\n".join(matched_texts)


# ==========================================================
# Step 2: LLM Judgement Per Clause
# ==========================================================

VALIDATOR_SYSTEM_PROMPT = """
أنت مراجع عقود قانوني مصري خبير.

هتاخد:
- اسم البند المطلوب ووصفه
- النص المستخرج من العقد المرفوع الخاص بهذا البند (أو "غير موجود")
- مرجع قانوني ذو صلة (لو متاح)

قيّم البند وارجع حكمك بصيغة JSON فقط بدون أي شرح خارجي:

{
  "status": "missing" أو "ok" أو "issue",
  "reason": "سبب مختصر جداً (سطر واحد بالعربي)"
}

قواعد الحكم:
- "missing": البند غير موجود في العقد نهائياً
- "ok": البند موجود وصياغته سليمة ومتوافقة مع القانون
- "issue": البند موجود لكن فيه مشكلة (مخالفة قانونية، صياغة غامضة، رقم غير منطقي، تعارض مع القانون المرجعي، ناقص تفاصيل جوهرية)

لا تخترع مشاكل غير موجودة. لو مفيش مرجع قانوني كافي للحكم، اعتبره "ok" ما لم يكن هناك مشكلة واضحة في النص نفسه.
"""

def judge_clause(clause: dict, section_text: str, law_context: str) -> dict:

    if section_text is None:
        return {
            "status": "missing",
            "reason": "البند غير موجود في نص العقد"
        }

    user_content = f"""
اسم البند: {clause['title']}
وصف البند: {clause.get('description', '')}

نص العقد الخاص بالبند:
{section_text}

مرجع قانوني ذو صلة:
{law_context if law_context else 'لا يوجد مرجع متاح'}
"""

    messages = [
        {"role": "system", "content": VALIDATOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0,
            max_tokens=250
        )

        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()

        result = json.loads(raw)

        if result.get("status") not in ("missing", "ok", "issue"):
            result["status"] = "ok"

        return result

    except Exception as e:
        print(f"Validator Error ({clause['title']}): {e}")
        return {
            "status": "ok",
            "reason": "تعذر التحقق الآلي من هذا البند"
        }


# ==========================================================
# Step 3: Full Contract Validation
# ==========================================================

def validate_contract(contract_type: str, sections: list) -> dict:
    """
    contract_type: "sale" | "lease" | "employment"
    sections: مخرجات contract_parser.py -> document["sections"]

    يرجع تقرير كامل عن كل required_clause
    """

    schema = load_schema(contract_type)

    if schema is None:
        return {"error": f"No schema found for {contract_type}"}

    report = []

    for clause in schema.get("required_clauses", []):

        section_text = find_clause_section(clause, sections)

        # نجيب مرجع قانوني بس لو البند موجود (عشان نقيّم مطابقته)
        # أو حتى لو ناقص، ممكن يفيد نعرف القانون بيقول إيه عنه
        law_query = clause["title"]
        law_hits = hybrid_retrieve("legal_docs", law_query, top_k=2)

        law_context = "\n".join(
            f"[{h['meta'].get('source','؟')} - مادة {h['meta'].get('article','؟')}] {h['text'][:300]}"
            for h in law_hits
        ) if law_hits else ""

        result = judge_clause(clause, section_text, law_context)

        report.append({
            "clause_id": clause["id"],
            "title": clause["title"],
            "status": result["status"],
            "reason": result["reason"]
        })

    summary = {
        "missing": sum(1 for r in report if r["status"] == "missing"),
        "issue": sum(1 for r in report if r["status"] == "issue"),
        "ok": sum(1 for r in report if r["status"] == "ok"),
    }

    return {
        "contract_type": contract_type,
        "summary": summary,
        "clauses": report
    }


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    from .contract_parser import parse_document
    from .contract_classifier import classify_contract

    pdf_path = r"E:\3th_2\NLP\legal_ai_assistant\data\صيغة عقد إيجار شقة سكنية.pdf"

    document = parse_document(pdf_path)

    classification = classify_contract(document["text"])
    contract_type = classification["contract_type"]

    print(f"Detected type: {contract_type} (method: {classification['method']})")
    
    result = validate_contract(contract_type, document["sections"])

    print(json.dumps(result, ensure_ascii=False, indent=2))