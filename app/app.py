# app/app.py
import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.analysis import analyze_video

UPLOAD_DIR = "/tmp/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Dance Movement Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    file_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        result = analyze_video(save_path, max_frames=600)  # limit frames for speed
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    # Optionally remove file after analysis
    try:
        os.remove(save_path)
    except Exception:
        pass

    return JSONResponse(content=result)
