from flask import Flask, Response, jsonify, request, send_from_directory
import cv2
import numpy as np
import threading
import base64
import json
import os
import time

# ── Optional: text-to-speech ──────────────────────────────────────────────────
try:
    from audio_engine import speak_text
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("[WARN] audio_engine not found – TTS disabled.")

# ── Optional: YOLO ────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO

    MODEL_PATHS = [
        os.path.join(os.path.dirname(__file__), "best.pt"),
        os.path.join(os.path.dirname(__file__), "yolov8n.pt"),
        os.path.join(os.path.dirname(__file__), "yolo26n.pt"),
        r"C:/WebAI/best.pt",
    ]
    model = None
    for path in MODEL_PATHS:
        if os.path.exists(path):
            print(f"[INFO] Loading model from {path}")
            model = YOLO(path)
            break
    if model is None:
        print(f"[WARN] Model not found. Tried: {MODEL_PATHS} – running without detection.")
except Exception as e:
    model = None
    print(f"[WARN] YOLO unavailable: {e}")


app = Flask(__name__, static_folder=".", template_folder=".")

# ── Shared state (protected by a lock) ───────────────────────────────────────
lock = threading.Lock()
state = {
    "run_camera": False,
    "sentence": "",
    "latest_char": "",
    "predicted_word": "Waiting...",
}

cap = None          
latest_frame = None 

# ─────────────────────────────────────────────────────────────────────────────
# Camera thread
# ─────────────────────────────────────────────────────────────────────────────
def camera_loop():
    global cap, latest_frame

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Failed to open camera. Check if camera is connected/available.")
        with lock:
            state["run_camera"] = False
        return

    print("[INFO] Camera opened successfully")

    while True:
        with lock:
            running = state["run_camera"]
        if not running:
            break

        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame from camera")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.flip(frame_rgb, 1)

        predicted_word = "Waiting..."
        current_detected_class = ""

        if model is not None:
            results = model(frame_rgb, conf=0.4, verbose=False)
            frame_h, frame_w, _ = frame_rgb.shape
            max_box_area = frame_h * frame_w * 0.4

            for r in results:
                boxes = r.boxes
                if len(boxes) == 0:
                    continue
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if (x2 - x1) * (y2 - y1) > max_box_area:
                        continue
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0]) * 100
                    current_detected_class = class_name
                    predicted_word = f"{class_name} ({confidence:.1f}%)"
                    cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame_rgb, predicted_word,
                        (x1, max(y1 - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2,
                    )
                    break  # only process biggest/first detection

        with lock:
            state["latest_char"] = current_detected_class
            state["predicted_word"] = predicted_word

        # Encode to JPEG for MJPEG stream
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        _, jpeg = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with lock:
            global latest_frame
            latest_frame = jpeg.tobytes()

    cap.release()
    cap = None
    print("[INFO] Camera closed")
    with lock:
        state["run_camera"] = False
        latest_frame = None

# ─────────────────────────────────────────────────────────────────────────────
# Routes – static files
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "index.html")
# ─────────────────────────────────────────────────────────────────────────────
# Routes – MJPEG video stream
# ─────────────────────────────────────────────────────────────────────────────
def generate_frames():
    """Generator that yields MJPEG frames."""
    while True:
        with lock:
            running = state["run_camera"]
            frame = latest_frame
        if not running or frame is None:
            # Send a blank placeholder so the <img> doesn't break
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + _blank_jpeg() + b"\r\n"
            import time; time.sleep(0.1)
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"


def _blank_jpeg():
    """1×1 grey JPEG used when the camera is off."""
    img = np.full((1, 1, 3), 50, dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )
# ─────────────────────────────────────────────────────────────────────────────
# Routes – state API
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/state")
def get_state():
    with lock:
        return jsonify({
            "run_camera": state["run_camera"],
            "sentence": state["sentence"],
            "latest_char": state["latest_char"],
            "predicted_word": state["predicted_word"],
        })


@app.route("/api/toggle_camera", methods=["POST"])
def toggle_camera():
    global _camera_thread
    with lock:
        state["run_camera"] = not state["run_camera"]
        running = state["run_camera"]

    if running:
        print("[INFO] Starting camera thread...")
        _camera_thread = threading.Thread(target=camera_loop, daemon=True)
        _camera_thread.start()
    else:
        print("[INFO] Stopping camera...")

    with lock:
        return jsonify({"run_camera": state["run_camera"]})

last_api_add_time = 0.0

@app.route("/api/confirm_char", methods=["POST"])
def confirm_char():
    global last_api_add_time
    with lock:
        current_time = time.time()
        
        # Chống gửi lệnh liên tục dưới 2 giây (Chống lag)
        if current_time - last_api_add_time < 2.0:
            return jsonify({"status": "spam_blocked"})
            
        last_api_add_time = current_time
        
        # Nhận chữ cái từ giao diện Web gửi lên
        data = request.get_json(silent=True)
        if data and "char" in data:
            char_to_add = data["char"]
        else:
            char_to_add = state["latest_char"]
        
        # Thêm chữ vào câu
        if char_to_add:
            if char_to_add == 'Space':
                state["sentence"] += " "
            elif char_to_add == 'Delete':
                state["sentence"] = state["sentence"][:-1]
            else:
                state["sentence"] += char_to_add
                
        return jsonify({
            "sentence": state["sentence"],
            "added_char": char_to_add,
            "status": "success"
        })

@app.route("/api/add_space", methods=["POST"])
def add_space():
    with lock:
        state["sentence"] += " "
        return jsonify({"sentence": state["sentence"]})

@app.route("/api/clear_all", methods=["POST"])
def clear_all():
    with lock:
        state["sentence"] = ""
        return jsonify({"sentence": state["sentence"]})

@app.route("/api/delete_char", methods=["POST"])
def delete_char():
    with lock:
        state["sentence"] = state["sentence"][:-1]
        return jsonify({"sentence": state["sentence"]})

@app.route("/api/speak", methods=["POST"])
def speak():
    with lock:
        text = state["sentence"].strip()
    if not text:
        return jsonify({"ok": False, "error": "Empty sentence"})
    if not TTS_AVAILABLE:
        return jsonify({"ok": False, "error": "TTS not available on server"})
    threading.Thread(target=speak_text, args=(text,), daemon=True).start()
    return jsonify({"ok": True})

# ─────────────────────────────────────────────────────────────────────────────
# Catch-all for static files (CSS, JS, etc.) – must come AFTER API routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/<path:filename>")
def static_files(filename):
    """Serve CSS, JS, and other static files from current directory."""
    if filename.endswith(('.css', '.js', '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico')):
        try:
            return send_from_directory(".", filename)
        except Exception as e:
            print(f"[WARN] Failed to serve {filename}: {e}")
            return "Not found", 404
    return "Not found", 404
# ─────────────────────────────────────────────────────────────────────────────
_camera_thread = None

if __name__ == "__main__":
    # Threaded=True so MJPEG stream and API calls don't block each other
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
