# ================================================================
# 🍓 Fruits Detection with YOLOv8 (Flask Version, Render Optimized)
# ================================================================

from flask import Flask, render_template, request, url_for
import os
from ultralytics import YOLO
from pathlib import Path
import gdown  # ✅ Used to download the model from Google Drive if not found

# ================================================================
# 🔧 Flask App Setup
# ================================================================
app = Flask(__name__)

# Create required folders for uploads and results
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ================================================================
# 🧠 Model Setup — Google Drive Download + Render Optimization
# ================================================================

# ✅ Google Drive model file ID (from your shared link)
file_id = "1IsF2SazGDWZjgwyAlBfKN2RBjug4aLeL"

# ✅ Direct download link for the model
model_url = f"https://drive.google.com/uc?id={file_id}"

# ✅ Download the model only if it doesn't exist locally
if not os.path.exists("best.pt"):
    print("⬇️ Downloading YOLOv8 model from Google Drive...")
    gdown.download(model_url, "best.pt", quiet=False)
else:
    print("✅ Model already exists — skipping download")

# ✅ Load YOLOv8 model
model = YOLO("best.pt")

# ✅ Disable layer fusion to prevent out-of-memory errors on Render
model.fuse = lambda *a, **k: model

# ================================================================
# 🏠 HOME PAGE — Landing Page
# ================================================================
@app.route('/')
def home():
    # This renders your 'home.html' template (mode selection)
    return render_template('home.html')

# ================================================================
# 📁 IMAGE UPLOAD PAGE — For Uploading Fruit Image
# ================================================================
@app.route('/upload_mode')
def upload_mode():
    # This renders 'index.html' (your upload page)
    return render_template('index.html')

# ================================================================
# 🍎 IMAGE UPLOAD DETECTION — Detection Logic
# ================================================================
@app.route('/upload', methods=['POST'])
def upload():
    # Check if file is uploaded properly
    if 'file' not in request.files:
        return "No file uploaded 😢"
    file = request.files['file']
    if file.filename == '':
        return "No selected file 😢"

    # ✅ Save uploaded image inside static/uploads folder
    upload_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(upload_path)

    # ✅ Run YOLOv8 model to detect fruits
    results = model.predict(
        source=upload_path,
        conf=0.5,
        save=True,
        project='static',   # base folder
        name='results',     # subfolder for results
        exist_ok=True       # overwrite if exists
    )

    # ✅ Locate the saved detected image inside static/results
    result_path = Path("static/results") / os.path.basename(file.filename)
    if not result_path.exists():
        # fallback in case YOLO renames output file
        detected_files = list(Path("static/results").glob("*.jpg")) + list(Path("static/results").glob("*.png"))
        if detected_files:
            result_path = max(detected_files, key=os.path.getctime)
        else:
            return "Detection completed but no image found 😭"

    # ✅ Convert image path to a Flask-accessible URL
    result_image_url = url_for('static', filename=f"results/{result_path.name}")

    # ================================================================
    # 🍉 Extract Detected Fruit Names and Add Emojis
    # ================================================================
    fruit_names = []
    for r in results:
        for c in r.boxes.cls:
            fruit_names.append(model.names[int(c)])
    fruit_names = list(set(fruit_names))  # remove duplicates

    # 🍎 Emoji mapping for detected fruits
    emojis = {
        'Apple': '🍎', 'Banana': '🍌', 'Carambola': '⭐',
        'Chilli': '🌶️', 'Coconut': '🥥', 'Dragon fruit': '🐉',
        'Black berry': '🫐', 'Fig': '🍈', 'Grapes': '🍇',
        'Lemon': '🍋', 'Lychee': '🍒', 'Papaya': '🥭',
        'Persimmon': '🍑', 'Pomegranate': '🍎',
        'Raspberry': '🍓', 'Tomato': '🍅'
    }

    # Combine fruit names with emojis (e.g., 🍎 Apple)
    fruit_display = [f"{emojis.get(f, '🍏')} {f}" for f in fruit_names]

    # ================================================================
    # 📸 Render Detection Result Page
    # ================================================================
    return render_template('result.html', image_file=result_image_url, fruits=fruit_display)

# ================================================================
# 🚀 Run Flask App (Local or Render)
# ================================================================
if __name__ == '__main__':
    app.run(debug=True)
