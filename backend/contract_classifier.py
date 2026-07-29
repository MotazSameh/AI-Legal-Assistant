import os
import json
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ==========================================================
# Paths
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONTRACTS_DIR = os.path.join(BASE_DIR, "data", "contracts")

CONTRACT_TYPES = ["sale", "lease", "employment"]

# ==========================================================
# Load Schemas (once)
# ==========================================================

def load_schema(contract_type):
    path = os.path.join(CONTRACTS_DIR, contract_type, f"{contract_type}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


_SCHEMAS_CACHE = {}

def get_all_schemas():
    if not _SCHEMAS_CACHE:
        for ct in CONTRACT_TYPES:
            schema = load_schema(ct)
            if schema:
                _SCHEMAS_CACHE[ct] = schema
    return _SCHEMAS_CACHE


# ==========================================================
# Pass 1: Keyword Scoring (from schema search_terms)
# ==========================================================

def score_by_keywords(text: str) -> dict:
    """
    بيرجع dict فيه score لكل contract_type بناءً على تكرار
    search_terms بتاعة كل clause في الـ schema داخل نص العقد
    """
    schemas = get_all_schemas()
    scores = {ct: 0 for ct in CONTRACT_TYPES}

    normalized_text = text

    for ct, schema in schemas.items():
        for clause in schema.get("required_clauses", []):
            for term in clause.get("search_terms", []):
                # نعد كل ظهور للمصطلح في النص
                count = normalized_text.count(term)
                scores[ct] += count

    return scores


def is_ambiguous(scores: dict, margin: int = 2) -> bool:
    """
    لو الفرق بين أعلى نتيجتين قليل، نعتبرها ambiguous ونحتاج LLM
    """
    sorted_scores = sorted(scores.values(), reverse=True)

    if len(sorted_scores) < 2:
        return False

    top, second = sorted_scores[0], sorted_scores[1]

    if top == 0:
        return True  # مفيش أي تطابق، محتاجين LLM يجتهد

    return (top - second) <= margin


# ==========================================================
# Pass 2: LLM Fallback
# ==========================================================

CLASSIFIER_SYSTEM_PROMPT = """
أنت مسئول فقط عن تحديد نوع العقد من نص عقد مصري.

الأنواع المتاحة فقط:
- "sale": عقد بيع
- "lease": عقد إيجار
- "employment": عقد عمل

اقرأ النص وحدد النوع الأقرب.

أخرج إجابتك بصيغة JSON فقط بدون أي شرح وبدون Markdown:

{
  "contract_type": "sale" أو "lease" أو "employment"
}
"""

def classify_by_llm(text: str) -> str:
    # بنبعت أول جزء من النص بس (كفاية للتصنيف وتوفير tokens)
    excerpt = text[:2000]

    messages = [
        {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": excerpt}
    ]

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0,
            max_tokens=50
        )

        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()

        result = json.loads(raw)
        contract_type = result.get("contract_type")

        if contract_type in CONTRACT_TYPES:
            return contract_type

    except Exception as e:
        print(f"Classifier LLM Error: {e}")

    return None


# ==========================================================
# Main Classify Function
# ==========================================================

def classify_contract(text: str) -> dict:
    """
    text: النص الكامل للعقد (بعد الـ parsing من contract_parser.py)

    يرجع:
    {
        "contract_type": "sale" | "lease" | "employment" | None,
        "method": "keywords" | "llm" | "failed",
        "scores": {...}   # للتشخيص/الـ debugging
    }
    """

    scores = score_by_keywords(text)

    if not is_ambiguous(scores):
        best_type = max(scores, key=scores.get)
        return {
            "contract_type": best_type,
            "method": "keywords",
            "scores": scores
        }

    # النتيجة مش واضحة -> نلجأ للـ LLM
    llm_result = classify_by_llm(text)

    if llm_result:
        return {
            "contract_type": llm_result,
            "method": "llm",
            "scores": scores
        }

    # فشل كل حاجة -> نرجع أعلى score حتى لو مش متأكدين
    fallback_type = max(scores, key=scores.get) if any(scores.values()) else None

    return {
        "contract_type": fallback_type,
        "method": "failed",
        "scores": scores
    }


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    sample_employment_text = """
    عقد عمل
    الطرف الأول: شركة النور للبرمجيات
    الطرف الثاني: السيد أحمد محمد
    يعمل الطرف الثاني بوظيفة مهندس برمجيات
    الأجر الشهري 8000 جنيه
    بدل انتقال 500 جنيه
    ساعات العمل ثماني ساعات يوميا
    فترة التجربة ثلاثة أشهر
    """

    result = classify_contract(sample_employment_text)
    print(json.dumps(result, ensure_ascii=False, indent=2))