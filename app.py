from flask import Flask, render_template, Response
import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime

app = Flask(__name__)

IMAGE_DIR = "images"
ATTENDANCE_FILE = "attendance.csv"
SIMILARITY_THRESHOLD = 0.55
CAMERA_INDEX = 0   # try 0 first, if external cam then 1

detector = cv2.FaceDetectorYN.create(
    "models/face_detection_yunet_2023mar.onnx", "", (320, 320)
)
recognizer = cv2.FaceRecognizerSF.create(
    "models/face_recognition_sface_2021dec.onnx", ""
)

known_names = []
known_rolls = []
known_embeddings = []

# ================= LOAD KNOWN FACES =================
for folder in os.listdir(IMAGE_DIR):
    path = os.path.join(IMAGE_DIR, folder)
    if not os.path.isdir(path):
        continue

    try:
        roll, name = folder.split("_", 1)
        roll = int(roll)
    except:
        continue

    person_embs = []

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
        person_embs.append(emb)

    if person_embs:
        avg_emb = np.mean(person_embs, axis=0)
        avg_emb = avg_emb / np.linalg.norm(avg_emb)
        known_embeddings.append(avg_emb)
        known_names.append(name)
        known_rolls.append(roll)

if not known_embeddings:
    print("No valid faces loaded")
    exit()

marked_today = set()

def mark_attendance(roll, name):
    today = datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.now().strftime("%H:%M:%S")

    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)
    else:
        df = pd.DataFrame(columns=["Roll No", "Name", "Date", "Time"])

    if not ((df["Roll No"] == roll) & (df["Date"] == today)).any():
        df.loc[len(df)] = [roll, name, today, time_now]
        df = df.sort_values("Roll No")
        df.to_csv(ATTENDANCE_FILE, index=False)
        print(f"Attendance marked: {name}")

# ================= CAMERA =================
cap = cv2.VideoCapture(1, cv2.CAP_MSMF)  # Use your external camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def gen_frames():
    global cap
    while True:
        success, frame = cap.read()
        if not success:
            break

        h, w = frame.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(frame)

        if faces is not None:
            for face in faces:
                x, y, fw, fh = map(int, face[:4])

                aligned = recognizer.alignCrop(frame, face)
                emb = recognizer.feature(aligned).flatten()
                emb = emb / np.linalg.norm(emb)

                sims = [np.dot(emb, ref) for ref in known_embeddings]
                idx = int(np.argmax(sims))
                score = sims[idx]

                if score > SIMILARITY_THRESHOLD:
                    name = known_names[idx]
                    roll = known_rolls[idx]

                    if roll not in marked_today:
                        marked_today.add(roll)
                        mark_attendance(roll, name)

                    label = f"{roll} - {name}"
                    color = (0, 255, 0)
                else:
                    label = "UNKNOWN"
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x+fw, y+fh), color, 2)
                cv2.putText(frame, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, threaded=False)
