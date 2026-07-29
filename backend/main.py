from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from backend.pipeline import handle_message          # ← بدل استيراد get_answer مباشرة
from urllib.parse import quote
from datetime import datetime
import os, json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================================================
# Paths
# ====================================================
PDF_PATH   = r"E:\3th_2\NLP\legal_ai_assistant\data"
LOGS_PATH  = r"E:\3th_2\NLP\legal_ai_assistant\logs"
UPLOAD_PATH = r"E:\3th_2\NLP\legal_ai_assistant\temp_uploads"   # ← جديد: مكان مؤقت لملفات العقود المرفوعة

os.makedirs(LOGS_PATH, exist_ok=True)
os.makedirs(UPLOAD_PATH, exist_ok=True)   # ← جديد

# ====================================================
# Models
# ====================================================
class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    answer: str
    sources: list = []
    flow: str = "legal"          # ← جديد: يوضح للفرونت إند هل الرد كان عن قانون ولا عقد
    extra: dict = {}             # ← جديد: أي بيانات إضافية (زي validation report) لو حابب تعرضها لاحقًا

# ====================================================
# Logging (زي ما هي، بدون أي تغيير)
# ====================================================
def save_log(session_id: str, question: str, answer: str, sources: list, flow: str = "legal", extra: dict = None):
    log_file = os.path.join(LOGS_PATH, f"{session_id}.json")

    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            log = json.load(f)
    else:
        log = []

    entry = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "flow": flow,
        "question": question,
        "answer": answer,
        "sources": sources
    }

    if extra:
        entry["extra"] = extra

    log.append(entry)

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

# ====================================================
# Helper: توحيد شكل الرد من pipeline.handle_message
# ====================================================
def normalize_result(result: dict) -> dict:
    """
    handle_message بترجع أشكال مختلفة حسب الـ flow (legal / contract / error).
    الدالة دي بتوحدهم في شكل واحد ثابت يترجع للفرونت إند.
    """

    flow = result.get("flow", "legal")

    if flow == "legal":
        return {
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "flow": "legal",
            "extra": {}
        }

    if flow == "contract":
        if result.get("error"):
            return {
                "answer": result.get("message", "حدث خطأ في معالجة العقد."),
                "sources": [],
                "flow": "contract_error",
                "extra": {"error": result.get("error")}
            }

        return {
            "answer": result.get("answer", ""),
            "sources": [],
            "flow": f"contract_{result.get('mode', 'unknown')}",   # contract_new_analysis أو contract_follow_up
            "extra": {
                "contract_type": result.get("contract_type"),
                "validation": result.get("validation")
            }
        }

    return {"answer": "حدث خطأ غير متوقع.", "sources": [], "flow": "error", "extra": {}}

# ====================================================
# Routes
# ====================================================
@app.get("/")
def home():
    return {"message": "Legal AI Assistant Running "}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    result = handle_message(req.session_id, req.message)
    normalized = normalize_result(result)

    save_log(
        req.session_id,
        req.message,                      # ← السؤال الحقيقي دايمًا
        normalized["answer"],
        normalized["sources"],
        flow=normalized["flow"],
        extra=normalized["extra"]
    )

    return ChatResponse(**normalized)


@app.post("/contract/upload", response_model=ChatResponse)
async def upload_contract(session_id: str = Form(...), file: UploadFile = File(...)):
    temp_path = os.path.join(UPLOAD_PATH, f"{session_id}_{file.filename}")

    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        result = handle_message(session_id, "راجع هذا العقد من فضلك", uploaded_file_path=temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    normalized = normalize_result(result)

    save_log(
        session_id,
        f"[رفع ملف: {file.filename}]",     # ← نضيف اسم الملف الفعلي كسياق
        normalized["answer"],
        [],
        flow=normalized["flow"],
        extra=normalized["extra"]           # ← دلوقتي التقرير الكامل هيتسجل
    )

    return ChatResponse(**normalized)

@app.delete("/chat/{session_id}")
def clear_history(session_id: str):
    import backend.session_store as session_store
    session_store.clear_session(session_id)
    return {"message": f"History cleared for session: {session_id}"}


@app.get("/pdf/{filename}")
def get_pdf(filename: str):
    filepath = os.path.join(PDF_PATH, filename)
    if not os.path.exists(filepath):
        return {"error": f"File not found: {filepath}"}
    encoded_name = quote(filename)
    return FileResponse(
        path=filepath,
        media_type="application/pdf",
        headers={"Content-Disposition": f"inline; filename*=UTF-8''{encoded_name}"}
    )

# e:/3th_2/NLP/legal_ai_assistant/venv/Scripts/uvicorn.exe backend.main:app --reload