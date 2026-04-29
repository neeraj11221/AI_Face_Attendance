---
title: AI Face Attendance System
emoji: 🎓
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
python_version: 3.10.13
app_file: app.py
pinned: false
license: mit
---

# 🎓 AI Face Attendance System

A real-time face recognition attendance system using **OpenCV YuNet + SFace** models — no cloud APIs, runs fully on-device.

## ✨ Features
- 📸 Live webcam face detection & recognition
- ✅ Auto attendance marking (once per student per day)
- ➕ Register new students with face images
- 📋 View & download attendance CSV

---

## 🚀 Full Local + Hugging Face Deployment Guide

### 1. Clone / Create your project folder

```
ai-face-attendance/
├── app.py
├── requirements.txt
├── download_models.py
├── models/              ← ONNX model files go here
│   ├── face_detection_yunet_2023mar.onnx
│   └── face_recognition_sface_2021dec.onnx
└── images/              ← Student face folders go here
    ├── 101_Neeraj_Sharma/
    │   ├── img1.jpg
    │   └── img2.jpg
    └── 102_Raj_Patel/
        └── img1.jpg
```

---

### 2. Download the ONNX models

Run this once in your project folder:

```bash
python download_models.py
```

This downloads `face_detection_yunet_2023mar.onnx` and `face_recognition_sface_2021dec.onnx` into `models/`.

---

### 3. Install dependencies locally

```bash
pip install -r requirements.txt
```

> ⚠️ On Windows: Use `opencv-contrib-python` instead of `opencv-contrib-python-headless` in requirements.txt

---

### 4. Run locally

```bash
python app.py
```

Open `http://127.0.0.1:7860` in your browser.

---

### 5. Register students

**Option A — Via the UI (Gradio)**
- Go to the **"Register Student"** tab
- Enter Roll No and Name
- Upload 3–5 clear face photos
- Click **Register Student**

**Option B — Manually**
Create folders inside `images/` using the format:
```
images/
└── ROLLNO_FirstName_LastName/
    ├── photo1.jpg
    ├── photo2.jpg
    └── photo3.jpg
```
Then click **Reload Student Database** in the app.

---

### 6. Deploy to Hugging Face Spaces

#### Step 1 — Install Git LFS (for large model files)
```bash
git lfs install
```

#### Step 2 — Create a new Space on huggingface.co
- Go to https://huggingface.co/new-space
- Name: `ai-face-attendance`
- SDK: **Gradio**
- Visibility: Public or Private

#### Step 3 — Clone your Space
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/ai-face-attendance
cd ai-face-attendance
```

#### Step 4 — Copy your files
Copy these into the cloned folder:
```
app.py
requirements.txt
download_models.py
README.md
models/   ← include both .onnx files
images/   ← include your registered student folders
```

#### Step 5 — Track model files with Git LFS
```bash
git lfs track "*.onnx"
git add .gitattributes
```

#### Step 6 — Commit and push
```bash
git add .
git commit -m "Initial deployment"
git push
```

Your Space will auto-build and deploy in 2–3 minutes. 🎉

---

## 📁 Image Folder Naming Convention

| Format | Example |
|--------|---------|
| `ROLLNO_Name` | `101_Neeraj` |
| `ROLLNO_First_Last` | `101_Neeraj_Sharma` |

Roll number must come **before** the first underscore.

---

## ⚙️ Threshold Guide

| Value | Behaviour |
|-------|-----------|
| 0.35 | Lenient — may cause false matches |
| 0.45 | Balanced ✅ (recommended) |
| 0.55 | Strict — may miss some faces |

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---------|-----|
| Camera not working on HF Space | HF Spaces support webcam via browser; allow camera permission |
| "No valid faces loaded" | Check folder names use `ROLL_Name` format |
| Model not found error | Run `download_models.py` and ensure `models/` folder is pushed |
| UNKNOWN for known student | Lower the threshold slider or add more training images |