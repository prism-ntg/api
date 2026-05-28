from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import text
from pydantic import BaseModel
from fastapi.responses import FileResponse

from database import get_db

MODEL_DIR = Path(__file__).parent / "models"

rf_model = joblib.load(MODEL_DIR / "prism_rf_model.pkl")
label_encoder = joblib.load(MODEL_DIR / "prism_label_encoder.pkl")
fitur_model = joblib.load(MODEL_DIR / "prism_features.pkl")

app = FastAPI(
    title="PRISM Inference Server",
    description="Predictive Reporting & Intelligent Scheduling Manager",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AssetFeatures(BaseModel):
    Kekritisan_Score: int
    Avg_Maintenance_Delay: float
    Max_Maintenance_Delay: float
    Total_Downtime: float
    Avg_Downtime: float
    Total_Biaya_Perbaikan: float
    Failure_Frequency: float
    Peak_Severity: float
    Avg_Biaya_Penggantian: float
    Cost_Risk_Ratio: float
    Umur_Aset_Hari: int


class BatchPredictRequest(BaseModel):
    data: list[AssetFeatures]


@app.get("/", tags=["Health"])
def root():
    return {
        "status": "ok",
        "message": "PRISM Inference Server berjalan",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        db_status = f"failed: {str(e)}"

    return {
        "status": "ok" if db_status == "connected" else "warning",
        "model_loaded": True,
        "db_status": db_status,
        "fitur": fitur_model,
        "kelas": list(label_encoder.classes_),
    }


@app.post("/predict", tags=["Prediksi"])
def predict(features: AssetFeatures):
    try:
        df = pd.DataFrame([features.model_dump()])
        df = df[fitur_model]
        pred_encoded = rf_model.predict(df)
        label = label_encoder.inverse_transform(pred_encoded)[0]
        return {"status": "success", "rekomendasi_jadwal": label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", tags=["Prediksi"])
def predict_batch(req: BatchPredictRequest):
    if not req.data:
        raise HTTPException(status_code=400, detail="Data tidak boleh kosong")
    try:
        hasil = []
        for item in req.data:
            df = pd.DataFrame([item.model_dump()])
            df = df[fitur_model]
            pred_encoded = rf_model.predict(df)
            proba = rf_model.predict_proba(df)[0]
            label = label_encoder.inverse_transform(pred_encoded)[0]
            hasil.append({
                "status": "success",
                "rekomendasi_jadwal": label,
                "probabilitas": {
                    cls: round(float(p), 4)
                    for cls, p in zip(label_encoder.classes_, proba)
                },
                "confidence": round(float(np.max(proba)), 4),
            })
        return {"total": len(hasil), "hasil": hasil}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/report/{filename}", tags=["Report"])
def get_report(filename: str):
    return FileResponse(path=f"./reports/{filename}", media_type="application/pdf")

class ChatRequest(BaseModel):
    pertanyaan: str

@app.post("/chat", tags=["AI Chat Bot"])
def chat(request: ChatRequest, db: Session = Depends(get_db), baseurl: str = "http://localhost:8000"):
    try:
        from nlp import process_nlp_report
        result = process_nlp_report(request.pertanyaan, db, baseurl)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/get-all-users", tags=["Database"])
def get_all_users(db: Session = Depends(get_db)):
    try:
        result = db.execute(text("SELECT * FROM users"))
        return [dict(row._mapping) for row in result.fetchall()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
