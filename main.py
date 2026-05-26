from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from database import engine, get_db

app = FastAPI()

@app.get("/")
def main(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        return {"message": "API is running", "db_status": "connected"}
    except Exception as e:
        return {"message": "API is running", "db_status": f"failed: {str(e)}"}

# cuma buat contoh cara pake database, db confignya samain sama yg di fe
@app.get("/get-all-users")
def get_all_users(db: Session = Depends(get_db)):
    result = db.execute(text("SELECT * FROM users"))
    users = [dict(row._mapping) for row in result.fetchall()]
    return users