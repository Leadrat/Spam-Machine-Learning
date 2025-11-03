# Spam Email Detector (FastAPI + React)

End-to-end web app to detect whether an email is Spam or Ham using NLP and an LSTM model.

## Folder Structure
```
spam-email-detector/
 ├── backend/
 │   ├── app.py
 │   ├── model/
 │   │   ├── train_model.py
 │   │   ├── spam_model.h5
 │   │   └── tokenizer.pkl
 │   ├── utils/
 │   │   ├── preprocess.py
 │   │   └── predict.py
 │   ├── requirements.txt
 │   └── README.md
 ├── frontend/
 │   ├── package.json
 │   ├── src/
 │   │   ├── App.jsx
 │   │   ├── components/
 │   │   │   ├── EmailForm.jsx
 │   │   │   ├── ResultCard.jsx
 │   │   └── styles.css
 │   ├── public/
 │   │   └── index.html
 │   └── README.md
 ├── dataset/Emails.csv
 ├── README.md
 └── .gitignore
```

## How to run

### 1) Backend
- Create venv and install deps
```
python -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt
```
- Train model (creates `backend/model/spam_model.h5` and `backend/model/tokenizer.pkl`)
```
python backend/model/train_model.py
```
- Run API (port 8000)
```
uvicorn backend.app:app --reload --port 8000
```

### 2) Frontend (Vite + React)
```
cd frontend
npm install
npm run dev
```
- Frontend dev server runs at http://localhost:5173
- Ensure `.env` in `frontend/` contains:
```
VITE_API_BASE_URL=http://localhost:8000
```

## API Example
POST http://localhost:8000/predict
Body:
```
{
  "text": "Congratulations, you've won a free lottery! Click now"
}
```
Response:
```
{
  "prediction": "Spam",
  "probability": 0.96
}
```

## Future Improvements
- Add word frequency charts (Chart.js or matplotlib)
- Persist user queries and predictions in SQLite
- Dockerize backend and frontend
