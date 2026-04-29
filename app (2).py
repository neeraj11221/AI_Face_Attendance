import gradio as gr
import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
import pytz
import threading
import time
import sys

# ================= SETTINGS =================
IMAGE_DIR       = "images"
ATTENDANCE_FILE = "attendance.csv"
IST             = pytz.timezone("Asia/Kolkata")
DETECT_WIDTH    = 320

# ================= LOAD MODELS =================
detector = cv2.FaceDetectorYN.create(
    "models/face_detection_yunet_2023mar.onnx", "", (320, 320)
)
recognizer = cv2.FaceRecognizerSF.create(
    "models/face_recognition_sface_2021dec.onnx", ""
)

# ================= GLOBAL STATE =================
known_data   = {}
marked_today = set()
_csv_lock    = threading.Lock()

# ── Background thread shared state ──
_latest_frame     = None
_latest_threshold = 0.45
_latest_result    = (None, "📷 Waiting for camera...")
_frame_lock       = threading.Lock()
_result_lock      = threading.Lock()
_camera_active    = False


# ================= LOAD KNOWN FACES =================
def load_known_faces():
    global known_data
    known_data.clear()
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
        return "⚠️ images/ folder not found. Add student folders and reload."

    count, skipped = 0, []
    for folder in sorted(os.listdir(IMAGE_DIR)):
        path = os.path.join(IMAGE_DIR, folder)
        if not os.path.isdir(path):
            continue
        parts = folder.split("_", 1)
        if len(parts) < 2:
            skipped.append(folder); continue
        roll_str, name = parts
        if not roll_str.isdigit():
            skipped.append(folder); continue

        roll, embeddings = int(roll_str), []
        for img_name in os.listdir(path):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img = cv2.imread(os.path.join(path, img_name))
            if img is None:
                continue
            h, w = img.shape[:2]
            detector.setInputSize((w, h))
            retval, faces = detector.detect(img)
            if faces is None or len(faces) == 0:
                continue
            try:
                aligned = recognizer.alignCrop(img, faces[0])
                emb     = recognizer.feature(aligned).flatten()
                emb     = emb / np.linalg.norm(emb)
                embeddings.append(emb)
            except Exception:
                continue

        if embeddings:
            avg              = np.mean(embeddings, axis=0)
            known_data[roll] = {"name": name, "embedding": avg / np.linalg.norm(avg)}
            count           += 1
        else:
            skipped.append(f"{folder} (no face detected)")

    msg = f"✅ Loaded {count} student(s)."
    if skipped:
        msg += f"\n⚠️ Skipped: {', '.join(skipped)}"
    return msg


# ================= MARK ATTENDANCE =================
def mark_attendance(roll, name):
    now_ist  = datetime.now(IST)
    today    = now_ist.strftime("%Y-%m-%d")
    time_now = now_ist.strftime("%H:%M:%S")

    with _csv_lock:
        df = pd.DataFrame(columns=["Roll No", "Name", "Date", "Time"])
        if os.path.exists(ATTENDANCE_FILE):
            try:
                df            = pd.read_csv(ATTENDANCE_FILE)
                df.dropna(how="all", inplace=True)
                df["Roll No"] = df["Roll No"].astype(str).str.replace(r"\.0$", "", regex=True)
            except Exception:
                pass

        safe_roll = str(roll).replace(".0", "")
        if not ((df["Roll No"] == safe_roll) & (df["Date"] == today)).any():
            new_row = pd.DataFrame([[safe_roll, name, today, time_now]],
                                   columns=["Roll No", "Name", "Date", "Time"])
            df = pd.concat([df.dropna(how="all"), new_row], ignore_index=True)
            df.to_csv(ATTENDANCE_FILE, index=False)
            return True
    return False


# ================= BACKGROUND DETECTION LOOP =================
def _detection_loop():
    global _latest_result
    # Own model instances — OpenCV DNN is not thread-safe across instances
    det = cv2.FaceDetectorYN.create(
        "models/face_detection_yunet_2023mar.onnx", "", (320, 320)
    )
    rec = cv2.FaceRecognizerSF.create(
        "models/face_recognition_sface_2021dec.onnx", ""
    )

    while True:
        with _frame_lock:
            frame     = _latest_frame
            threshold = _latest_threshold

        if frame is None or not _camera_active:
            time.sleep(0.05)
            continue

        try:
            orig_h, orig_w = frame.shape[:2]
            scale = min(1.0, DETECT_WIDTH / orig_w)
            small = cv2.resize(frame, (int(orig_w * scale), int(orig_h * scale))) \
                    if scale < 1.0 else frame

            bgr  = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
            h, w = bgr.shape[:2]
            det.setInputSize((w, h))
            retval, faces = det.detect(bgr)

            bgr_full  = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            log_lines = []

            if faces is not None and len(faces) > 0:
                for face in faces:
                    x, y, fw, fh = map(int, face[:4])
                    if scale < 1.0:
                        x, y   = int(x / scale), int(y / scale)
                        fw, fh = int(fw / scale), int(fh / scale)
                    x  = max(0, x);  y  = max(0, y)
                    fw = min(fw, orig_w - x);  fh = min(fh, orig_h - y)

                    try:
                        aligned = rec.alignCrop(bgr, face)
                        emb     = rec.feature(aligned).flatten()
                        emb     = emb / np.linalg.norm(emb)
                    except Exception:
                        continue

                    best_score, best_roll = -1.0, None
                    for roll, info in known_data.items():
                        sim = float(np.dot(emb, info["embedding"]))
                        if sim > best_score:
                            best_score, best_roll = sim, roll

                    if best_score >= threshold and best_roll is not None:
                        name  = known_data[best_roll]["name"]
                        color = (0, 255, 0)
                        label = f"{best_roll}-{name} ({best_score:.2f})"
                        if best_roll not in marked_today:
                            if mark_attendance(best_roll, name):
                                log_lines.append(f"✅ Marked: {name} (Roll {best_roll})")
                            marked_today.add(best_roll)
                        else:
                            log_lines.append(f"👤 Present: {name}")
                    else:
                        color = (0, 0, 255)
                        label = f"UNKNOWN ({best_score:.2f})" if best_roll else "UNKNOWN"

                    cv2.rectangle(bgr_full, (x, y), (x + fw, y + fh), color, 2)
                    cv2.putText(bgr_full, label, (x, max(y - 10, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                log_lines.append("👁️ No face detected.")

            out_rgb = cv2.cvtColor(bgr_full, cv2.COLOR_BGR2RGB)
            log     = "\n".join(log_lines) if log_lines else "👁️ Scanning..."
            with _result_lock:
                _latest_result = (out_rgb, log)

        except Exception:
            pass

        time.sleep(0.1)   # ~10 fps detection


_bg_thread = threading.Thread(target=_detection_loop, daemon=True)
_bg_thread.start()


# ================= STREAM CALLBACK — returns instantly =================
def process_frame(frame, threshold):
    global _latest_frame, _latest_threshold, _camera_active, _latest_result

    if frame is None or not isinstance(frame, np.ndarray) \
            or frame.size == 0 or frame.shape[0] == 0:
        _camera_active = False
        with _frame_lock:
            _latest_frame = None
        with _result_lock:
            _latest_result = (None, "📷 Camera stopped.")
        return None, "📷 Camera stopped."

    _camera_active = True
    with _frame_lock:
        _latest_frame     = frame
        _latest_threshold = threshold

    with _result_lock:
        return _latest_result


# ================= ATTENDANCE HELPERS =================
def get_attendance():
    with _csv_lock:
        if os.path.exists(ATTENDANCE_FILE):
            try:
                df = pd.read_csv(ATTENDANCE_FILE)
                df.dropna(how="all", inplace=True)
                if not df.empty:
                    df["Roll No"] = df["Roll No"].astype(str).str.replace(r"\.0$", "", regex=True)
                    return df.values.tolist()
            except Exception:
                pass
    return []

def download_attendance():
    return ATTENDANCE_FILE if os.path.exists(ATTENDANCE_FILE) else None

def delete_attendance():
    global marked_today
    with _csv_lock:
        if os.path.exists(ATTENDANCE_FILE):
            os.remove(ATTENDANCE_FILE)
    marked_today.clear()
    return []


# ================= INIT =================
startup_msg = load_known_faces()
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
print(startup_msg)


# ================= GRADIO UI =================
with gr.Blocks(title="AI Face Attendance System") as demo:

    gr.Markdown("# 🎓 AI Face Attendance System")
    gr.Markdown("**YuNet detection · SFace recognition · OpenCV DNN**")

    with gr.Tabs():

        with gr.TabItem("📸 Mark Attendance"):
            with gr.Row():
                with gr.Column(scale=1):
                    cam_input = gr.Image(
                        sources=["webcam"], streaming=True,
                        type="numpy", label="📷 Webcam Input", height=360
                    )
                with gr.Column(scale=1):
                    cam_output = gr.Image(
                        type="numpy", label="🟩 Detection Output", height=360
                    )

            threshold_slider = gr.Slider(
                minimum=0.30, maximum=0.70, value=0.45, step=0.01,
                label="Recognition Threshold  (lower = easier match | higher = stricter)"
            )
            log_box    = gr.Textbox(label="Status Log", lines=4, interactive=False)
            reload_btn = gr.Button("🔄 Reload Student Database", variant="secondary")
            reload_out = gr.Textbox(label="Database Status", interactive=False, lines=2)

            cam_input.stream(
                fn=process_frame,
                inputs=[cam_input, threshold_slider],
                outputs=[cam_output, log_box],
                api_name=False,
            )
            reload_btn.click(fn=load_known_faces, outputs=reload_out, api_name=False)

        with gr.TabItem("📋 View Attendance"):
            gr.Markdown("### Attendance Records")
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh Table",      variant="primary")
                dl_btn      = gr.Button("⬇️ Download CSV",       variant="secondary")
                delete_btn  = gr.Button("🗑️ Delete All Records", variant="stop")

            att_table = gr.Dataframe(
                value=[], headers=["Roll No", "Name", "Date", "Time"],
                type="array", label="Attendance Log",
                interactive=False, wrap=True
            )
            dl_file = gr.File(label="Download Ready", visible=True)

            refresh_btn.click(fn=get_attendance,   outputs=att_table, api_name=False)
            dl_btn.click(fn=download_attendance,   outputs=dl_file,   api_name=False)
            delete_btn.click(fn=delete_attendance, outputs=att_table, api_name=False)

    demo.load(fn=get_attendance, outputs=att_table, api_name=False)

if __name__ == "__main__":
    demo.queue().launch(share=True)