import os
from .intent_router import classify_intent
from .contract_parser import parse_document
from .contract_classifier import classify_contract
from .find_clause_section import validate_contract
from .contract_chat import generate_report_summary, discuss_contract
from .chatbot import get_answer
from . import session_store
# ==========================================================
# Unified Pipeline
# ==========================================================

def handle_message(session_id: str, query: str, uploaded_file_path: str = None):
    """
    query: رسالة المستخدم النصية
    chat_history: تاريخ المحادثة (بنفس شكل chatbot.py)
    uploaded_file_path: مسار ملف PDF لو المستخدم رفع عقد، وإلا None

    يرجع dict موحد لكل السيناريوهات:
    {
        "flow": "legal" | "contract",
        ... باقي التفاصيل حسب النوع
    }
    """

    has_file = uploaded_file_path is not None
    session = session_store.get_session(session_id)
    # ------------------------------------------------------
    # 1) Intent Routing
    # ------------------------------------------------------

    routing = classify_intent(query, has_uploaded_file=has_file)

    intent = routing["intent"]
    contract_type_hint = routing.get("contract_type")

    # ------------------------------------------------------
    # 2A) Legal Flow
    # ------------------------------------------------------

    if intent == "legal":
        legal_history = session["legal_chat_history"]
        answer, updated_history, sources = get_answer(query, legal_history)

        session_store.update_legal_chat_history(session_id, updated_history)

        return {
            "flow": "legal",
            "answer": answer,
            "sources": sources
        }

    # ------------------------------------------------------
    # 2B) Contract Flow
    # ------------------------------------------------------

    # حالة: مفيش ملف جديد لكن فيه عقد سابق في السيشن -> يبقى ده سؤال متابعة
    existing_contract = session_store.get_contract_state(session_id)


    if not has_file:
        if existing_contract is None:
            return {
                "flow": "contract",
                "error": "no_file",
                "message": "من فضلك ارفع ملف العقد (PDF) عشان أقدر أراجعه."
            }

        # سؤال متابعة عن آخر عقد اتحلل
        answer, updated_history = discuss_contract(
            query=query,
            contract_type=existing_contract["contract_type"],
            validation_report=existing_contract["validation_report"],
            sections=existing_contract["sections"],
            chat_history=existing_contract["chat_history"]
        )

        session_store.update_contract_chat_history(session_id, updated_history)

        return {
            "flow": "contract",
            "mode": "follow_up",
            "answer": answer
        }
    # ------------------------------------------------------
    # عقد جديد اتبعت -> نعمل التحليل من الأول
    # ------------------------------------------------------

    document = parse_document(uploaded_file_path)

    if document is None:
        return {
            "flow": "contract",
            "error": "parse_failed",
            "message": "تعذر قراءة الملف المرفوع، تأكد إنه PDF سليم."
        }

    # 2.2) Classify (لو الـ Router قدر يحدد النوع من الكلام، نستخدمه على طول ونوفر LLM call)
    if contract_type_hint:
        contract_type = contract_type_hint
        classify_method = "router"
    else:
        classification = classify_contract(document["text"])
        contract_type = classification["contract_type"]
        classify_method = classification["method"]

    if contract_type is None:
        return {
            "flow": "contract",
            "error": "unknown_type",
            "message": "تعذر تحديد نوع العقد، من فضلك حدد هل هو عقد بيع/إيجار/عمل."
        }

    # 2.3) Validate
    validation_report = validate_contract(contract_type, document["sections"])

    summary_text = generate_report_summary(contract_type, validation_report)
    # نحفظ حالة العقد ده في الـ session عشان أي سؤال متابعة يستخدمها
    session_store.save_contract_state(
        session_id,
        contract_type,
        validation_report,
        document["sections"]
    )
    return {
        "flow": "contract",
        "mode": "new_analysis",
        "contract_type": contract_type,
        "classify_method": classify_method,
        "validation": validation_report,
        "answer": summary_text
    }

# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    session_id = "test_user_1"

    # 1) رفع عقد جديد
    pdf_path = r"E:\3th_2\NLP\legal_ai_assistant\data\نموذج عقد بيع ابتدائي.pdf"
    result1 = handle_message(session_id, "راجعلي العقد ده", uploaded_file_path=pdf_path)
    print("=== First Analysis ===")
    print(result1["answer"])

    # 2) سؤال متابعة عن نفس العقد (من غير رفع ملف تاني)
    result2 = handle_message(session_id, "فيه بند ناقص إيه بالظبط؟")
    print("\n=== Follow-up ===")
    print(result2["answer"])

    # 3) سؤال قانوني عام (منفصل تمامًا)
    result3 = handle_message(session_id, "ما هي عقوبة السرقة؟")
    print("\n=== Legal Question ===")
    print(result3["answer"][:200])