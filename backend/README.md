# Backend (FastAPI)

## Setup
```
python -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt
```

## Train the model
```
python backend/model/train_model.py
```
This will create `backend/model/spam_model.h5` and `backend/model/tokenizer.pkl`.

## Run the API
```
uvicorn backend.app:app --reload --port 8000
```
Open http://localhost:8000/docs to try the Swagger UI.
