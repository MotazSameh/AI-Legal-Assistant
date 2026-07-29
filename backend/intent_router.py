import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ====================================================
# Intent Router
# ====================================================
"""
بيحدد نوع السؤال:
- "legal"    -> سؤال قانوني عام (المستخدم عاوز يسأل عن مادة/قانون)
- "contract" -> المستخدم عاوز يحلل/يسأل عن عقد (سواء رافع ملف أو بيسأل عنه نظريًا)

بيرجع dict فيها:
{
    "intent": "legal" | "contract",
    "contract_type": "sale" | "lease" | "employment" | null   # لو intent = contract ومقدر يحدد النوع من كلام المستخدم
}
"""

ROUTER_SYSTEM_PROMPT = """
أنت مسئول فقط عن تصنيف رسالة المستخدم داخل نظام قانوني مصري.

صنّف الرسالة إلى واحدة فقط من الفئتين:

1) "legal": المستخدم بيسأل سؤال قانوني عام (عن مادة، حكم، عقوبة، حق، إجراء قانوني...) وليس عن عقد محدد رفعه أو عاوز يكتبه.

2) "contract": المستخدم بيسأل عن عقد (بيع/إيجار/عمل) سواء:
   - رفع ملف عقد وعاوز يحلله أو يراجعه
   - عاوز يكتب/يفهم بنود عقد معين
   - بيسأل عن نوع عقد بالتحديد (بيع - إيجار - عمل)

أخرج إجابتك بصيغة JSON فقط بدون أي كلام إضافي وبدون Markdown:

{
  "intent": "legal" أو "contract",
  "contract_type": "sale" أو "lease" أو "employment" أو null
}

قواعد تحديد contract_type:
- لو عقد بيع / شراء -> "sale"
- لو عقد إيجار / تأجير -> "lease"
- لو عقد عمل / توظيف -> "employment"
- لو intent = "legal" أو مش واضح النوع -> null

لا تشرح. لا تضف أي نص خارج الـ JSON.
"""

def classify_intent(query: str, has_uploaded_file: bool = False) -> dict:
    """
    query: رسالة المستخدم
    has_uploaded_file: هل المستخدم رفع ملف PDF فعليًا في نفس الرسالة
    """

    user_content = query

    if has_uploaded_file:
        user_content += "\n\n[ملاحظة: المستخدم رفع ملف عقد مع الرسالة]"

    messages = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0,
            max_tokens=100
        )

        raw = response.choices[0].message.content.strip()

        # تنظيف لو رجع الموديل الجواب جوا ```json ... ```
        raw = raw.replace("```json", "").replace("```", "").strip()

        result = json.loads(raw)

        intent = result.get("intent")
        contract_type = result.get("contract_type")

        if intent not in ("legal", "contract"):
            intent = "legal"  # fallback آمن

        if contract_type not in ("sale", "lease", "employment"):
            contract_type = None

        return {
            "intent": intent,
            "contract_type": contract_type,
            "has_uploaded_file": has_uploaded_file
        }

    except Exception as e:
        print(f"Router Error: {e}")
        # fallback: لو المستخدم رفع ملف فعلي، الأرجح إنه عاوز تحليل عقد
        return {
            "intent": "contract" if has_uploaded_file else "legal",
            "contract_type": None,
            "has_uploaded_file": has_uploaded_file
        }


# ====================================================
# Test
# ====================================================

if __name__ == "__main__":

    tests = [
        ("ما هي عقوبة السرقة في القانون المصري؟", False),
        ("عاوز أراجع عقد الإيجار اللي رفعته", True),
        ("ايه شروط عقد العمل الصحيح؟", False),
        ("رفعت عقد بيع سيارة وعاوز أتأكد إنه سليم", True),
    ]

    for query, has_file in tests:
        result = classify_intent(query, has_file)
        print(f"\nQuery: {query}")
        print(f"Result: {result}")