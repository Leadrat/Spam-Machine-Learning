from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import json
from datetime import datetime
from pathlib import Path
from backend.utils.predict import load_artifacts, predict_text, explain_text
try:
    # Optional Gmail utilities (fetch only)
    from spam_email_detector.src.gmail_fetch import fetch_latest_emails
except Exception:
    fetch_latest_emails = None

# Optional Gemini (Generative AI) setup
_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
_gemini_model = None
try:
    if _GEMINI_API_KEY:
        import google.generativeai as genai
        genai.configure(api_key=_GEMINI_API_KEY)
        _gemini_model = genai.GenerativeModel("gemini-1.5-flash")
except Exception as _e:
    _gemini_model = None

app = FastAPI(title="Spam Email Detector API")

# CORS for all origins for simplicity in dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    text: str
    
class ChatRequest(BaseModel):
    message: str
    maxEmails: Optional[int] = 5
    sessionId: Optional[str] = None

class ChatResponse(BaseModel):
    type: str
    message: Optional[str] = None
    isSpam: Optional[bool] = None
    probability: Optional[float] = None
    reason: Optional[str] = None
    advice: Optional[str] = None
    topTokens: Optional[List[Dict[str, Any]]] = None
    items: Optional[List[Dict[str, Any]]] = None
    llmMessage: Optional[str] = None

@app.on_event("startup")
def _load():
    # Preload model/tokenizer on startup for faster inferences
    try:
        load_artifacts()
    except Exception as e:
        # Continue startup; first prediction can still attempt lazy load
        print(f"[Startup] Warning: {e}")

@app.get("/")
def root():
    return {"status": "ok", "message": "Spam Email Detector API"}

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        label, prob = predict_text(req.text)
        return {"prediction": label, "probability": round(float(prob), 4)}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model artifacts missing: {e}. Please run training script.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


HELP_TEXT = (
    "I use a Logistic Regression classifier with TF-IDF features trained on labeled emails. "
    "I convert email text into word frequency vectors, then compute a spam probability. "
    "Ask me to analyze text, explain results, or (optionally) analyze your recent Gmail emails."
)

EXAMPLE_SPAM = (
    "Congratulations! You have won a FREE prize. Click here to claim your gift now."
)

ANTI_SPAM_TIPS = (
    "Be cautious of unsolicited offers, check sender addresses, avoid clicking unknown links, and never share sensitive information."
)

EXAMPLES_SPAM_LIST = [
    "Claim your $1000 gift card now! Click here.",
    "URGENT: Your account will be suspended. Verify immediately.",
    "You are selected as a winner! Free iPhone, limited time offer.",
]
EXAMPLES_HAM_LIST = [
    "Meeting reminder: Project sync at 4PM today.",
    "Invoice attached for last month’s services.",
    "Your OTP for login is 482193.",
]


def _intent_from_message(msg: str) -> str:
    m = (msg or "").strip().lower()
    if any(k in m for k in ["how does this work", "how do you work", "what model", "help"]):
        return "help"
    if any(k in m for k in ["example spam", "spam example", "sample spam", "show example spam", "show spam examples", "spam examples"]):
        return "example_spam"
    if any(k in m for k in ["example ham", "ham example", "sample ham", "not spam example", "show ham examples", "ham examples"]):
        return "example_ham"
    if any(k in m for k in ["example", "examples", "samples", "show examples", "show sample"]):
        return "example_generic"
    if any(k in m for k in ["anti-spam", "tips", "safety tips"]):
        return "tips"
    if any(k in m for k in ["gmail", "my emails", "unread emails", "analyze my last"]):
        return "gmail"
    return "predict"


def _log_chat(payload: Dict[str, Any]):
    try:
        logs_dir = Path(__file__).resolve().parents[0] / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        with open(logs_dir / "chat_logs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _llm_reply(system_hint: str, user_message: str, model_result: Optional[Dict[str, Any]]) -> Optional[str]:
    if _gemini_model is None:
        return None
    try:
        summary = ""
        if model_result:
            prob = model_result.get("probability")
            prob_pct = f"{(prob*100):.2f}%" if isinstance(prob, (int, float)) else ""
            is_spam = "Spam" if model_result.get("isSpam") else "Not Spam"
            reason = model_result.get("reason") or ""
            advice = model_result.get("advice") or ""
            summary = f"\nModel: {is_spam} ({prob_pct}). {reason} {advice}".strip()
        prompt = (
            f"You are a helpful spam-detection assistant. {system_hint}\n"
            f"User message: {user_message}\n"
            f"{summary}"
        )
        resp = _gemini_model.generate_content(prompt)
        return (resp.text or "").strip()
    except Exception:
        return None


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        intent = _intent_from_message(req.message)
        timestamp = datetime.utcnow().isoformat()
        if intent == "help":
            resp = ChatResponse(type="info", message=HELP_TEXT)
            _log_chat({"t": timestamp, "sessionId": req.sessionId, "intent": intent, "user": req.message, "resp": resp.model_dump()})
            return resp
        if intent in ("example_spam", "example_generic"):
            # Return a few spam examples; for generic requests show both types in message
            msg = "Here are some spam examples:" if intent == "example_spam" else "Here are some examples. First, spam:" 
            items = [{"text": t, "label": "Spam"} for t in EXAMPLES_SPAM_LIST]
            # If generic, also include ham below
            if intent == "example_generic":
                items += [{"text": t, "label": "Ham"} for t in EXAMPLES_HAM_LIST]
            resp = ChatResponse(type="examples", items=items, message=msg)
            _log_chat({"t": timestamp, "sessionId": req.sessionId, "intent": intent, "user": req.message, "resp": resp.model_dump()})
            return resp
        if intent == "example_ham":
            resp = ChatResponse(type="examples", items=[{"text": t, "label": "Ham"} for t in EXAMPLES_HAM_LIST], message="Here are some non-spam examples:")
            _log_chat({"t": timestamp, "sessionId": req.sessionId, "intent": intent, "user": req.message, "resp": resp.model_dump()})
            return resp
        if intent == "tips":
            resp = ChatResponse(type="info", message=ANTI_SPAM_TIPS)
            _log_chat({"t": timestamp, "sessionId": req.sessionId, "intent": intent, "user": req.message, "resp": resp.model_dump()})
            return resp
        if intent == "gmail":
            if fetch_latest_emails is None:
                resp = ChatResponse(type="error", message="Gmail integration is not configured in this environment.")
                _log_chat({"t": timestamp, "sessionId": req.sessionId, "intent": intent, "user": req.message, "resp": resp.model_dump()})
                return resp
            emails = fetch_latest_emails(max_results=int(req.maxEmails or 5))
            items = []
            for e in emails:
                # Classify using the same backend model artifacts
                subj = e.get("subject", "") or "(No subject)"
                body = e.get("body", "") or ""
                text = f"{subj}\n\n{body}".strip()
                label, prob = predict_text(text)
                items.append({
                    "subject": subj,
                    "isSpam": str(label).lower().startswith("spam"),
                    "probability": float(prob)
                })
            resp = ChatResponse(type="gmail_summary", items=items, message=f"Analyzed {len(items)} recent emails.")
            _log_chat({"t": timestamp, "sessionId": req.sessionId, "intent": intent, "user": req.message, "resp": resp.model_dump()})
            return resp

        # default: prediction/explanation for provided message
        exp = explain_text(req.message)
        exp["probability"] = round(float(exp["probability"]), 4)
        llm = _llm_reply("Summarize the classification in friendly tone and provide 1-2 actionable tips.", req.message, exp)
        resp = ChatResponse(**exp, llmMessage=llm)
        _log_chat({"t": timestamp, "sessionId": req.sessionId, "intent": intent, "user": req.message, "resp": resp.model_dump()})
        return resp
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model artifacts missing: {e}. Please run training script.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
