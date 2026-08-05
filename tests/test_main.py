import sys
import os
# Menambahkan folder utama (api) ke dalam sistem path agar Python bisa menemukan 'main.py'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "message": "PRISM Inference Server berjalan",
        "docs": "/docs",
    }

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["model_loaded"] is True
    # Kita tidak terlalu ketat soal status DB (bisa connected atau warning)
    # yang penting endpointnya berjalan dan modelnya dimuat.
    assert "status" in data

def test_predict_endpoint():
    """Mengetes fungsionalitas model Machine Learning melalui endpoint /predict"""
    payload = {
        "id_aset": "TEST-ASET-01",
        "Kekritisan_Score": 3,
        "Avg_Maintenance_Delay": 1.5,
        "Max_Maintenance_Delay": 4.0,
        "Total_Downtime": 10.0,
        "Avg_Downtime": 2.5,
        "Total_Biaya_Perbaikan": 500000.0,
        "Failure_Frequency": 4.0,
        "Peak_Severity": 3.0,
        "Avg_Biaya_Penggantian": 150000.0,
        "Cost_Risk_Ratio": 1.2,
        "Umur_Aset_Hari": 365
    }
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["id_aset"] == "TEST-ASET-01"
    assert "rekomendasi_jadwal" in data

import pytest

@pytest.mark.skip(reason="Test ini memanggil API Groq betulan. Di GitHub Actions akan gagal. Gunakan test_chat_mocked saja.")
def test_chat_endpoint():
    """Mengetes fungsionalitas fitur Chat NLP"""
    payload = {
        "pertanyaan": "bagaimana cara merawat aset ini?"
    }
    response = client.post("/chat", json=payload)
    
    assert response.status_code == 200
    # Karena respon NLP bentuknya bervariasi (bergantung pada model/LLM),
    # kita cukup mengecek bahwa API merespons dengan sesuatu (bukan null)
    assert response.json() is not None

# ==========================================
# MISI 1: NEGATIVE TESTING (SKENARIO GAGAL)
# ==========================================

def test_predict_negative_missing_field():
    """Menguji apa yang terjadi jika data tidak lengkap dikirim ke /predict"""
    payload = {
        "id_aset": "TEST-ASET-02",
        # Sengaja menghilangkan field penting seperti 'Kekritisan_Score' dsb
    }
    response = client.post("/predict", json=payload)
    
    # FastAPI otomatis menolak data yang tidak sesuai skema (skema Pydantic)
    # Status HTTP 422 berarti "Unprocessable Entity" (Data tidak bisa diproses)
    assert response.status_code == 422
    
    # Memastikan FastAPI memberikan penjelasan error di field "detail"
    error_data = response.json()
    assert "detail" in error_data

def test_chat_negative_empty_payload():
    """Menguji endpoint /chat dengan payload kosong"""
    response = client.post("/chat", json={})
    
    # Harusnya ditolak (422) karena field 'pertanyaan' itu wajib (required)
    assert response.status_code == 422

# ==========================================
# MISI 2: MOCKING & STUBBING (AI PALSU)
# ==========================================
from unittest.mock import patch

def test_chat_mocked():
    """Menguji endpoint /chat tanpa memanggil model AI betulan (Mengehemat waktu & biaya)"""
    payload = {
        "pertanyaan": "pertanyaan testing"
    }
    
    # Kita "cegat" (patch) fungsi process_nlp_report yang ada di file nlp.py
    # agar saat API dipanggil, fungsi aslinya TIDAK berjalan.
    with patch("nlp.process_nlp_report") as mock_nlp:
        # Kita tentukan balasan pura-puranya (Stubbing)
        mock_nlp.return_value = {"jawaban": "Ini adalah jawaban dari AI bohongan!"}
        
        # Robot menembak API
        response = client.post("/chat", json=payload)
        
        # Pastikan status 200 OK
        assert response.status_code == 200
        data = response.json()
        
        # Pastikan balasannya adalah balasan bohongan kita!
        assert data["jawaban"] == "Ini adalah jawaban dari AI bohongan!"
        
        # mencoba memanggil AI (walaupun sudah dicegat) sebanyak 1 kali
        mock_nlp.assert_called_once()

# ==========================================
# MISI 3: PYTEST FIXTURES (SETUP & TEARDOWN)
# ==========================================
import pytest
from unittest.mock import MagicMock
from database import get_db

# `@pytest.fixture` adalah tanda bahwa fungsi ini bertugas menyiapkan sesuatu sebelum test berjalan,
# dan membersihkannya setelah test selesai.
@pytest.fixture
def mock_db_session():
    # --- FASE 1: SETUP (Berjalan SEBELUM test dimulai) ---
    # Bayangkan ini adalah proses membuat database "In-Memory" (sementara) yang kosong.
    mock_session = MagicMock()
    
    # Kita pura-pura isi database tersebut dengan 1 user dummy
    mock_row = MagicMock()
    mock_row._mapping = {"id": 999, "name": "User Dummy Rahasia"}
    mock_result = MagicMock()
    mock_result.fetchall.return_value = [mock_row]
    mock_session.execute.return_value = mock_result
    
    # Trik andalan FastAPI: Kita paksa aplikasi memakai database palsu ini!
    app.dependency_overrides[get_db] = lambda: mock_session
    
    # Memberikan database palsu ini ke fungsi test yang membutuhkannya
    yield mock_session
    
    # --- FASE 2: TEARDOWN (Berjalan otomatis SETELAH test selesai) ---
    # Biasanya dipakai untuk menghapus data dummy atau menutup koneksi
    app.dependency_overrides.clear()

# Perhatikan parameter `mock_db_session`. Dengan memasukkan nama fixture ke parameter,
# Pytest akan otomatis menjalankan Setup sebelum masuk ke dalam fungsi ini.
def test_get_all_users_with_fixture(mock_db_session):
    """Mencoba endpoint /get-all-users dengan jaminan tidak merusak database asli"""
    response = client.get("/get-all-users")
    
    assert response.status_code == 200
    data = response.json()
    
    # Pastikan data yang didapat HANYA data dari database bohongan kita
    assert len(data) == 1
    assert data[0]["name"] == "User Dummy Rahasia"
