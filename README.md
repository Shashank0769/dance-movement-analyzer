# Dance Movement Analyzer

**Dance Movement Analyzer** is a lightweight FastAPI service that analyzes short dance videos, extracts body keypoints using **MediaPipe** + **OpenCV**, applies rule-based pose classification (e.g., *Floss Step*, *Hands Up*, *Squat*, *T-pose*), and returns a JSON summary with per-frame labels and aggregated counts. The project is containerized with Docker and ready to deploy to any Docker-friendly cloud (Railway, Render, AWS, etc.).

---

## Table of Contents
- [Features](#features)  
- [Project Structure](#project-structure)  
- [Requirements](#requirements)  
- [Local Setup (Windows / Linux / macOS)](#local-setup-windows--linux--macos)  
- [Run Tests](#run-tests)  
- [Run Locally (without Docker)](#run-locally-without-docker)  
- [Docker (build & run)](#docker-build--run)  
- [Deploy to Railway (quick)](#deploy-to-railway-quick)  
- [API Usage](#api-usage)  
- [Response Format](#response-format)  
- [Configuration & Environment Variables](#configuration--environment-variables)  
- [Troubleshooting (common issues)](#troubleshooting-common-issues)  
- [Demo Recording Checklist (2-minute demo)](#demo-recording-checklist-2-minute-demo)  
- [Security & Production Notes](#security--production-notes)  
- [Potential Improvements](#potential-improvements)  
- [License](#license)

---

## Features
- Uses **MediaPipe Pose** to extract 33 body landmarks per frame.
- Rule-based pose classification.
- FastAPI endpoint `/analyze` that accepts video upload.
- Per-frame labels and aggregated pose counts in JSON.
- Dockerized for easy deployment.
- Unit tests with `pytest`.

---

## Project Structure
```
dance-movement-analyzer/
├─ app/
│  ├─ app.py
│  ├─ analysis.py
│  └─ pose_utils.py
├─ tests/
│  ├─ test_pose_utils.py
│  └─ test_analysis_synthetic.py
├─ demo_sample_videos/
├─ requirements.txt
├─ Dockerfile
├─ README.md
└─ .gitignore
```

---

## Requirements
- Python **3.10**
- Docker Desktop
- `pip`

`requirements.txt` contains:
```
fastapi
uvicorn[standard]
opencv-python-headless
mediapipe
numpy
pydantic
pytest
python-multipart
requests
```

---

## Local Setup (Windows / Linux / macOS)

### 1. Create Virtual Environment
```bash
python3.10 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Run Tests
```bash
python -m pytest -q
```

---

## Run Locally (without Docker)
```bash
uvicorn app.app:app --reload --host 0.0.0.0 --port 8000
```
Then open: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Docker (build & run)

```bash
docker build -t dance-movement-analyzer .
docker run -p 8000:8000 dance-movement-analyzer
```

---

## Deploy to Railway (quick)

1. Push to GitHub:
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-username>/dance-movement-analyzer.git
git push -u origin main
```

2. Go to [https://railway.app](https://railway.app)

3. New Project → Deploy from GitHub Repo → Select Repo

4. Railway auto-detects Dockerfile and deploys.

5. Access your app at:  
   `https://<your-app>.up.railway.app/docs`

---

## API Usage

**Endpoint**
```
POST /analyze
Content-Type: multipart/form-data
Form field name: file
```

**Example**
```bash
curl -X POST "https://<your-app>.up.railway.app/analyze"   -F "file=@/path/to/demo.mp4;type=video/mp4"
```

---

## Response Format
```json
{
  "total_frames": 208,
  "fps": 30,
  "pose_counts": {
    "Floss Step": 40,
    "Neutral Pose": 138
  },
  "sample_frames": [
    {"time_s": 0, "labels": ["Neutral Pose"]},
    {"time_s": 1.36, "labels": ["Floss Step"]}
  ]
}
```

---

## Troubleshooting
- Restart Docker if daemon issues occur.
- Use Python 3.10 for MediaPipe.
- Add `libgl1` & `libglib2.0-0` to Docker image if OpenCV errors appear.

---

## Demo Checklist
1. Show file structure.
2. Start container or show Railway URL.
3. Open `/docs` and upload sample video.
4. Show JSON response.
5. Highlight `pose_counts`.

---
