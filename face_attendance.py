import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime

IMAGE_DIR = "images"
ATTENDANCE_FILE = "attendance.csv"
SIMILARITY_THRESHOLD = 0.45
CAMERA_INDEX = 1

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
    person_path = os.path.join(IMAGE_DIR, folder)
    if not os.path.isdir(person_path):
        continue

    roll_no, name = folder.split("_", 1)

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)
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

        known_names.append(name)
        known_rolls.append(int(roll_no))
        known_embeddings.append(emb)

if len(known_embeddings) == 0:
    print("No valid faces loaded")
    exit()

# ================= ATTENDANCE FUNCTION =================
from datetime import datetime
import pandas as pd
import os

def mark_attendance(roll, name):
    today = datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.now().strftime("%H:%M:%S")

    file = "attendance.csv"

    if os.path.exists(file):
        df = pd.read_csv(file)
    else:
        df = pd.DataFrame(columns=["Roll No", "Name", "Date", "Time"])

    # Ensure columns exist
    if "Roll No" not in df.columns:
        df["Roll No"] = ""
    if "Name" not in df.columns:
        df["Name"] = ""
    if "Date" not in df.columns:
        df["Date"] = ""
    if "Time" not in df.columns:
        df["Time"] = ""

    # 🔒 CHECK: already marked today?
    already_marked = ((df["Roll No"] == roll) & (df["Date"] == today)).any()

    if not already_marked:
        new_row = pd.DataFrame([[roll, name, today, time_now]],
                               columns=["Roll No", "Name", "Date", "Time"])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(file, index=False)
        print(f"✅ Attendance marked for {name}")
    else:
        print(f"⚠ Attendance already marked for {name}")

# ================= CAMERA =================
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

marked_today = set()

while True:
    ret, frame = cap.read()
    if not ret:
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
            best_idx = int(np.argmax(sims))
            best_score = sims[best_idx]

            if best_score > SIMILARITY_THRESHOLD:
                name = known_names[best_idx]
                roll = known_rolls[best_idx]

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

    cv2.imshow("Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()