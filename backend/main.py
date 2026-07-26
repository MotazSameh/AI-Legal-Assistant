from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from backend.chatbot import get_answer
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
PDF_PATH  = r"E:\3th_2\NLP\legal_ai_assistant\data"
LOGS_PATH = r"E:\3th_2\NLP\legal_ai_assistant\logs"
os.makedirs(LOGS_PATH, exist_ok=True)  #  بيعمل الفولدر لو مش موجود

# ====================================================
# Sessions
# ====================================================
sessions = {}

# ====================================================
# Models
# ====================================================
class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    answer: str
    history: list
    sources: list

# ====================================================
# Logging
# ====================================================
def save_log(session_id: str, question: str, answer: str, sources: list):
    log_file = os.path.join(LOGS_PATH, f"{session_id}.json")

    # لو الملف موجود حمّله، لو لأ ابدأ list فاضية
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            log = json.load(f)
    else:
        log = []

    log.append({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "answer": answer,
        "sources": sources
    })

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

# ====================================================
# Routes
# ====================================================
@app.get("/")
def home():
    return {"message": "Legal AI Assistant Running "}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if req.session_id not in sessions:
        sessions[req.session_id] = []

    history = sessions[req.session_id]
    answer, updated_history, sources = get_answer(req.message, history)
    sessions[req.session_id] = updated_history

    #  احفظ الـ log
    save_log(req.session_id, req.message, answer, sources)

    return ChatResponse(answer=answer, history=updated_history, sources=sources)


@app.delete("/chat/{session_id}")
def clear_history(session_id: str):
    sessions.pop(session_id, None)
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

#e:/3th_2/NLP/legal_ai_assistant/venv/Scripts/uvicorn.exe backend.main:app --reload
