from flask import Flask, render_template, request
import cv2
import os
import numpy as np
from datetime import datetime
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from PIL import Image

app = Flask(__name__)

# Folder to store results
UPLOAD_FOLDER = 'static/results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# --- Simulated Deep Learning Enhancer (CNN/GAN-like effect) ---
def deep_learning_enhancement(img):
    enhanced = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
    enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
    return enhanced


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/enhance', methods=['POST'])
def enhance():
    if 'file' not in request.files:
        return "⚠️ No file uploaded!"

    file = request.files['file']
    if file.filename == '':
        return "⚠️ No selected file!"

    # 🔹 Create unique filename
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{timestamp}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # 🔹 Read image safely (works for all formats)
    img = None
    try:
        # Try normal OpenCV read first
        img = cv2.imread(filepath)

        # If that fails, try decoding bytes (handles Unicode names)
        if img is None:
            with open(filepath, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)

        # If still None, use Pillow for unsupported formats (HEIC, WEBP, etc.)
        if img is None:
            pil_img = Image.open(filepath).convert('RGB')
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    except Exception as e:
        return f"⚠️ Error reading image: {e}"

    if img is None:
        return f"⚠️ Unable to read image! File: {file.filename}"

    # --- Step 1: Resize ---
    resized = cv2.resize(img, (256, 256))

    # --- Step 2: Classical Enhancement ---
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    hist_eq = cv2.equalizeHist(gray)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_eq = clahe.apply(gray)

    # --- Step 3: Deep Learning Simulation ---
    deep_eq = deep_learning_enhancement(resized)

    # --- Step 4: Compute Metrics ---
    psnr_hist = psnr(gray, hist_eq)
    ssim_hist = ssim(gray, hist_eq)
    psnr_clahe = psnr(gray, clahe_eq)
    ssim_clahe = ssim(gray, clahe_eq)
    psnr_deep = psnr(gray, cv2.cvtColor(deep_eq, cv2.COLOR_BGR2GRAY))
    ssim_deep = ssim(gray, cv2.cvtColor(deep_eq, cv2.COLOR_BGR2GRAY))

    # --- Step 5: Save outputs ---
    hist_path = f"hist_{timestamp}.jpg"
    clahe_path = f"clahe_{timestamp}.jpg"
    deep_path = f"deep_{timestamp}.jpg"
    resized_path = f"resized_{timestamp}.jpg"

    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], resized_path), resized)
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], hist_path), hist_eq)
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], clahe_path), clahe_eq)
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], deep_path), deep_eq)

    # --- Step 6: Display Results ---
    return render_template(
        'result.html',
        original=resized_path,
        hist=hist_path,
        clahe=clahe_path,
        deep=deep_path,
        psnr_hist=round(psnr_hist, 2),
        ssim_hist=round(ssim_hist, 2),
        psnr_clahe=round(psnr_clahe, 2),
        ssim_clahe=round(ssim_clahe, 2),
        psnr_deep=round(psnr_deep, 2),
        ssim_deep=round(ssim_deep, 2)
    )


if __name__ == '__main__':
    app.run(debug=True)
