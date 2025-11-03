from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.utils.predict import load_artifacts, predict_text

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
