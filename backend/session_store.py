# ==========================================================
# session_store.py
# ==========================================================
"""
تخزين بسيط في الذاكرة لحالة كل جلسة (session).
كل session_id بيحمل:
- legal_chat_history: تاريخ أسئلة القانون العامة
- contract: آخر عقد اتحلل (النوع + التقرير + الـ sections + تاريخ نقاشه)
"""

_sessions = {}


def get_session(session_id: str) -> dict:
    """
    يرجع الـ session لو موجودة، أو ينشئ واحدة جديدة فاضية
    """
    if session_id not in _sessions:
        _sessions[session_id] = {
            "legal_chat_history": [],
            "contract": None   # لسه مفيش عقد اتحلل
        }
    return _sessions[session_id]


def save_contract_state(session_id: str, contract_type: str, validation_report: dict, sections: list):
    """
    بتتنادى بعد ما عقد جديد يتحلل، تحفظ كل حاجة محتاجينها للنقاش بعد كده
    """
    session = get_session(session_id)
    session["contract"] = {
        "contract_type": contract_type,
        "validation_report": validation_report,
        "sections": sections,
        "chat_history": []   # نقاش خاص بالعقد ده لوحده
    }


def get_contract_state(session_id: str) -> dict:
    """
    يرجع آخر عقد اتحلل في السيشن دي، أو None لو مفيش
    """
    session = get_session(session_id)
    return session["contract"]


def update_contract_chat_history(session_id: str, chat_history: list):
    session = get_session(session_id)
    if session["contract"] is not None:
        session["contract"]["chat_history"] = chat_history


def update_legal_chat_history(session_id: str, chat_history: list):
    session = get_session(session_id)
    session["legal_chat_history"] = chat_history


def clear_session(session_id: str):
    _sessions.pop(session_id, None)