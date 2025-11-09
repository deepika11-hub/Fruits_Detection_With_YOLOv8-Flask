from flask import Flask, render_template, request, url_for, Response
import os
from ultralytics import YOLO
from pathlib import Path
import cv2

# ======================================================
# 🔧 Flask App Setup
# ======================================================
app = Flask(__name__)

# Static folder paths
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load YOLOv8 model
model = YOLO("best.pt")

# ======================================================
# 🏠 HOME PAGE — Choose Mode
# ======================================================
@app.route('/')
def home():
    return render_template('home.html')

# ======================================================
# 📁 IMAGE UPLOAD MODE PAGE
# ======================================================
@app.route('/upload_mode')
def upload_mode():
    return render_template('index.html')

# ======================================================
# 🎥 WEBCAM PAGE
# ======================================================
@app.route('/webcam')
def webcam():
    return render_template('webcam.html')


# ======================================================
# 🔴 LIVE VIDEO STREAM (Webcam Detection)
# ======================================================
def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Run YOLOv8 model on webcam frame
            results = model.predict(source=frame, conf=0.5, stream=True)
            for r in results:
                annotated_frame = r.plot()  # Draw bounding boxes
                _, buffer = cv2.imencode('.jpg', annotated_frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
# ======================================================
# 🍎 IMAGE UPLOAD DETECTION (Your Existing Working Logic)
# ======================================================
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file uploaded 😢"
    file = request.files['file']
    if file.filename == '':
        return "No selected file 😢"

    # Save uploaded image
    upload_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(upload_path)

    # Run YOLOv8 and save results to static/results
    results = model.predict(
        source=upload_path,
        conf=0.5,
        save=True,
        project='static',
        name='results',
        exist_ok=True
    )

    # Find result image path
    result_path = Path("static/results") / os.path.basename(file.filename)
    if not result_path.exists():
        detected_files = list(Path("static/results").glob("*.jpg")) + list(Path("static/results").glob("*.png"))
        if detected_files:
            result_path = max(detected_files, key=os.path.getctime)
        else:
            return "Detection completed but no image found 😭"

    result_image_url = url_for('static', filename=f"results/{result_path.name}")

    # Extract detected fruit names
    fruit_names = []
    for r in results:
        for c in r.boxes.cls:
            fruit_names.append(model.names[int(c)])
    fruit_names = list(set(fruit_names))

    # Add fruit emojis 🍎🍌🍇
    emojis = {
        'Apple': '🍎', 'Banana': '🍌', 'Carambola': '⭐',
        'Chilli': '🌶️', 'Coconut': '🥥', 'Dragon fruit': '🐉',
        'Black berry': '🫐', 'Fig': '🍈', 'Grapes': '🍇',
        'Lemon': '🍋', 'Lychee': '🍒', 'Papaya': '🥭',
        'Persimmon': '🍑', 'Pomegranate': '🍎',
        'Raspberry': '🍓', 'Tomato': '🍅'
    }

    fruit_display = [f"{emojis.get(f, '🍏')} {f}" for f in fruit_names]

    return render_template('result.html', image_file=result_image_url, fruits=fruit_display)

# ======================================================
# 🚀 Run Flask App
# ======================================================
if __name__ == '__main__':
    app.run(debug=True)
