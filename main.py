from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import text
from pydantic import BaseModel, Field
from datetime import datetime

from database import engine, get_db

# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------
MODEL_DIR = Path(__file__).parent / "models"
ml: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml["model"] = joblib.load(MODEL_DIR / "prism_rf_model.pkl")
    ml["encoder"] = joblib.load(MODEL_DIR / "prism_label_encoder.pkl")
    ml["features"] = joblib.load(MODEL_DIR / "prism_features.pkl")
    yield
    ml.clear()


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="PRISM API",
    description="API prediksi jadwal perawatan aset + database integration",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    aset_id: str | None = None
    kekritisan_score: int = Field(..., ge=1, le=3)
    avg_maintenance_delay: float = Field(..., ge=0)
    max_maintenance_delay: float = Field(..., ge=0)
    total_downtime: float = Field(..., ge=0)
    avg_downtime: float = Field(..., ge=0)
    total_biaya_perbaikan: float = Field(..., ge=0)
    failure_frequency: int = Field(..., ge=0)
    peak_severity: float = Field(..., ge=0)
    avg_biaya_penggantian: float = Field(..., ge=0)
    cost_risk_ratio: float = Field(..., ge=0)
    umur_aset_hari: int = Field(..., ge=0)


class PredictResponse(BaseModel):
    aset_id: str | None = None
    rekomendasi_jadwal: str
    probabilitas: dict[str, float]
    confidence: float


class BatchPredictRequest(BaseModel):
    data: list[PredictRequest]


class BatchPredictResponse(BaseModel):
    hasil: list[PredictResponse]
    total: int


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _to_dataframe(req: PredictRequest) -> pd.DataFrame:
    row = {
        "Kekritisan_Score": req.kekritisan_score,
        "Avg_Maintenance_Delay": req.avg_maintenance_delay,
        "Max_Maintenance_Delay": req.max_maintenance_delay,
        "Total_Downtime": req.total_downtime,
        "Avg_Downtime": req.avg_downtime,
        "Total_Biaya_Perbaikan": req.total_biaya_perbaikan,
        "Failure_Frequency": req.failure_frequency,
        "Peak_Severity": req.peak_severity,
        "Avg_Biaya_Penggantian": req.avg_biaya_penggantian,
        "Cost_Risk_Ratio": req.cost_risk_ratio,
        "Umur_Aset_Hari": req.umur_aset_hari,
    }
    return pd.DataFrame([row], columns=ml["features"])


def _predict_single(req: PredictRequest) -> PredictResponse:
    df = _to_dataframe(req)
    pred_encoded = ml["model"].predict(df)
    proba = ml["model"].predict_proba(df)[0]
    label = ml["encoder"].inverse_transform(pred_encoded)[0]
    classes = ml["encoder"].classes_
    confidence = float(np.max(proba))

    return PredictResponse(
        aset_id=req.aset_id,
        rekomendasi_jadwal=label,
        probabilitas={cls: round(float(p), 4) for cls, p in zip(classes, proba)},
        confidence=round(confidence, 4),
    )


# ---------------------------------------------------------------------------
# Health & Status Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["Health"])
def root():
    return {
        "status": "ok",
        "message": "PRISM API berjalan",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        db_status = f"failed: {str(e)}"

    model_loaded = "model" in ml
    return {
        "status": "ok" if (model_loaded and db_status == "connected") else "warning",
        "model_loaded": model_loaded,
        "db_status": db_status,
        "fitur": ml.get("features", []),
        "kelas": list(ml["encoder"].classes_) if model_loaded else [],
    }


# ---------------------------------------------------------------------------
# ML Prediction Endpoints
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=PredictResponse, tags=["Prediksi"])
def predict(req: PredictRequest):
    """Prediksi jadwal perawatan untuk satu aset."""
    try:
        return _predict_single(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Prediksi"])
def predict_batch(req: BatchPredictRequest):
    """Prediksi jadwal perawatan untuk banyak aset."""
    if not req.data:
        raise HTTPException(status_code=400, detail="Data tidak boleh kosong")
    try:
        hasil = [_predict_single(item) for item in req.data]
        return BatchPredictResponse(hasil=hasil, total=len(hasil))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Database Endpoints (contoh)
# ---------------------------------------------------------------------------
@app.get("/", tags=["Database"])
def main(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        return {"message": "API is running", "db_status": "connected"}
    except Exception as e:
        return {"message": "API is running", "db_status": f"failed: {str(e)}"}


@app.get("/get-all-users", tags=["Database"])
def get_all_users(db: Session = Depends(get_db)):
    result = db.execute(text("SELECT * FROM users"))
    users = [dict(row._mapping) for row in result.fetchall()]
    return users