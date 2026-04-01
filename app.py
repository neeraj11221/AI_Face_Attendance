import streamlit as st
import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
import time

# ================= SETTINGS =================
IMAGE_DIR = "images"
ATTENDANCE_FILE = "attendance.csv"
CAMERA_INDEX = 0

# ================= LOAD MODELS =================
@st.cache_resource
def load_models():
    detector = cv2.FaceDetectorYN.create(
        "models/face_detection_yunet_2023mar.onnx", "", (320, 320)
    )
    recognizer = cv2.FaceRecognizerSF.create(
        "models/face_recognition_sface_2021dec.onnx", ""
    )
    return detector, recognizer

detector, recognizer = load_models()

# ================= LOAD FACES =================
@st.cache_resource
def load_faces():
    data = {}

    for folder in os.listdir(IMAGE_DIR):
        path = os.path.join(IMAGE_DIR, folder)
        if not os.path.isdir(path):
            continue

        roll, name = folder.split("_", 1)
        embeddings = []

        for img_name in os.listdir(path):
            img = cv2.imread(os.path.join(path, img_name))
            if img is None:
                continue

            h, w = img.shape[:2]
            detector.setInputSize((w, h))
            _, faces = detector.detect(img)

            if faces is None:
                continue

            aligned = recognizer.alignCrop(img, faces[0])
            emb = recognizer.feature(aligned).flatten()
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)

        if embeddings:
            avg = np.mean(embeddings, axis=0)
            avg = avg / np.linalg.norm(avg)
            data[int(roll)] = {"name": name, "embedding": avg}

    return data

known_data = load_faces()

# ================= ATTENDANCE =================
def mark_attendance(roll, name):
    today = datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.now().strftime("%H:%M:%S")

    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)
    else:
        df = pd.DataFrame(columns=["Roll No", "Name", "Date", "Time"])

    if not ((df["Roll No"] == roll) & (df["Date"] == today)).any():
        new_row = pd.DataFrame([[roll, name, today, time_now]],
                               columns=["Roll No", "Name", "Date", "Time"])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)
        return True
    return False

# ================= STREAMLIT UI =================
st.title("Face Attendance System")

threshold = st.slider("Accuracy Threshold", 0.3, 0.7, 0.45)

col1, col2 = st.columns(2)
if col1.button("Start Camera"):
    st.session_state.run = True

if col2.button("Stop Camera"):
    st.session_state.run = False

if "run" not in st.session_state:
    st.session_state.run = False

if "cap" not in st.session_state:
    st.session_state.cap = cv2.VideoCapture(CAMERA_INDEX)

if "marked" not in st.session_state:
    st.session_state.marked = set()

FRAME = st.empty()
status = st.empty()

# ================= FRAME PROCESS =================
def process_frame(frame):
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)

    if faces is not None:
        for face in faces:
            x, y, fw, fh = map(int, face[:4])

            aligned = recognizer.alignCrop(frame, face)
            emb = recognizer.feature(aligned).flatten()
            emb = emb / np.linalg.norm(emb)

            best_score = -1
            best_roll = None

            for roll, info in known_data.items():
                sim = np.dot(emb, info["embedding"])
                if sim > best_score:
                    best_score = sim
                    best_roll = roll

            if best_score > threshold:
                name = known_data[best_roll]["name"]

                if best_roll not in st.session_state.marked:
                    if mark_attendance(best_roll, name):
                        status.success(f"Marked: {name}")
                    st.session_state.marked.add(best_roll)

                label = f"{best_roll}-{name}"
                color = (0, 255, 0)
            else:
                label = "UNKNOWN"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+fw, y+fh), color, 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return frame

# ================= MAIN LOOP (FIXED) =================
if st.session_state.run:
    cap = st.session_state.cap

    ret, frame = cap.read()
    if ret:
        frame = process_frame(frame)
        FRAME.image(frame, channels="BGR")
    else:
        st.error("Camera not working")

    time.sleep(0.03)
    st.rerun()

# ================= TABLE =================
st.subheader("Attendance")

if os.path.exists(ATTENDANCE_FILE):
    df = pd.read_csv(ATTENDANCE_FILE)
    st.dataframe(df)

    st.download_button(
        "Download CSV",
        df.to_csv(index=False),
        file_name="attendance.csv"
    )