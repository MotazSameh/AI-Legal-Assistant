import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ==========================================================
# System Prompt: تحويل التقرير لرد طبيعي + نقاش مفتوح بعد كده
# ==========================================================

CONTRACT_CHAT_SYSTEM_PROMPT = """
أنت مساعد قانوني مصري متخصص في مراجعة العقود.

لديك تقرير تحليل آلي لعقد قام المستخدم برفعه، ونص العقد الأصلي مقسّم لبنود.

قواعد صارمة:
- أجب فقط بناءً على التقرير ونص العقد المرفقين، لا تخترع أي معلومة
- لو سُئلت عن بند مش موجود في التقرير، قل "غير مذكور في التحليل المتاح"
- كن مختصرًا وواضحًا، بدون تكرار
- لو فيه بنود "missing"، وضّح للمستخدم إنها لازم تتضاف
- لو فيه بنود "issue"، اشرح المشكلة بوضوح واقترح تعديل لو ممكن
- صحح أي خطأ إملائي في كلامك تلقائيًا
"""

def build_report_context(contract_type: str, validation_report: dict) -> str:
    """
    بيحول التقرير الخام (JSON) لنص منظم يتبعت كـ context للـ LLM
    """

    summary = validation_report["summary"]

    lines = [
        f"نوع العقد: {contract_type}",
        f"الملخص: {summary['ok']} بند سليم، {summary['missing']} بند ناقص، {summary['issue']} بند فيه مشكلة",
        "",
        "تفاصيل البنود:"
    ]

    for clause in validation_report["clauses"]:
        status_ar = {
            "ok": "سليم",
            "missing": "ناقص",
            "issue": "فيه مشكلة"
        }.get(clause["status"], clause["status"])

        line = f"- {clause['title']}: {status_ar}"
        if clause["reason"]:
            line += f" ({clause['reason']})"

        lines.append(line)

    return "\n".join(lines)


# ==========================================================
# First Message: تحويل التقرير لرد نصي مباشرة بعد التحليل
# ==========================================================

def generate_report_summary(contract_type: str, validation_report: dict) -> str:
    """
    بتتنادى مرة واحدة بعد validate_contract مباشرة
    عشان تطلع أول رد نصي طبيعي للمستخدم
    """

    report_context = build_report_context(contract_type, validation_report)

    messages = [
        {"role": "system", "content": CONTRACT_CHAT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"""
{report_context}

اكتب ملخص طبيعي وواضح لهذا التحليل للمستخدم، يوضح فيه:
- حالة العقد بشكل عام
- البنود الناقصة (لو فيه)
- البنود اللي فيها مشكلة وليه (لو فيه)
"""
        }
    ]

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0,
        max_tokens=600
    )

    return response.choices[0].message.content


# ==========================================================
# Follow-up Questions: نقاش مفتوح بعد الرد الأول
# ==========================================================

def discuss_contract(
    query: str,
    contract_type: str,
    validation_report: dict,
    sections: list,
    chat_history: list
) -> tuple[str, list]:
    """
    query: سؤال المستخدم المتابعة (زي "ليه بند الأجر فيه مشكلة؟")
    validation_report: نفس التقرير اللي طلع من validate_contract (بنحتفظ بيه في session)
    sections: sections العقد الأصلية (من contract_parser.py) - لو المستخدم سأل عن تفاصيل مش في الملخص
    chat_history: history خاص بالمحادثة دي عن العقد ده (منفصل عن legal chat history)

    يرجع: (answer, updated_history) - بنفس شكل get_answer في chatbot.py
    """

    report_context = build_report_context(contract_type, validation_report)

    # نص العقد الكامل (كل الـ sections) كمرجع لو المستخدم سأل عن تفاصيل دقيقة
    full_contract_text = "\n---\n".join(s["content"] for s in sections)

    messages = [{"role": "system", "content": CONTRACT_CHAT_SYSTEM_PROMPT}]
    messages += chat_history

    messages.append({
        "role": "user",
        "content": f"""
تقرير التحليل:
{report_context}

نص العقد الكامل (للرجوع إليه عند الحاجة):
{full_contract_text}

سؤال المستخدم: {query}
"""
    })

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0,
        max_tokens=600,
        presence_penalty=0.15
    )

    answer = response.choices[0].message.content

    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": answer})

    if len(chat_history) > 10:
        chat_history = chat_history[-10:]

    return answer, chat_history