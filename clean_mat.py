import cv2
import numpy as np
import os
from sklearn.cluster import MiniBatchKMeans

# --- CONFIGURATION ---
INPUT_FOLDER = 'scans'
OUTPUT_FOLDER = 'cleaned_auto_v3'

# We increase this to 16 to ensure the Green Outline 
# doesn't get merged into the Yellow Fill.
NUM_COLORS = 16 

# UPSCALING PRESERVES TEXT
# 3x means we make the image 3 times bigger before processing.
SCALE_FACTOR = 3

# ---------------------

def quantize_colors(image, n_colors):
    """
    Standard K-Means color reduction (Auto-Color).
    """
    (h, w) = image.shape[:2]
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image_lab = image_lab.reshape((image_lab.shape[0] * image_lab.shape[1], 3))

    clt = MiniBatchKMeans(n_clusters=n_colors, n_init=10, batch_size=4096)
    labels = clt.fit_predict(image_lab)
    quant = clt.cluster_centers_.astype("uint8")[labels]

    quant = quant.reshape((h, w, 3))
    return cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)

def process_image(filename):
    img_path = os.path.join(INPUT_FOLDER, filename)
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Could not load {filename}")
        return

    print(f"Processing {filename}...")
    h, w = img.shape[:2]

    # 1. UPSCALE (The Secret Weapon)
    # We make the image massive. Now, a "speck of dust" is still small,
    # but a "letter" is HUGE.
    print("  - Upscaling...")
    img_large = cv2.resize(img, (w * SCALE_FACTOR, h * SCALE_FACTOR), interpolation=cv2.INTER_CUBIC)

    # 2. STRONG SMOOTHING (From your original script)
    # This flattens the paper texture.
    print("  - Smoothing...")
    smooth = cv2.bilateralFilter(img_large, 9, 100, 100)

    # 3. AUTO-COLOR (K-Means)
    # This respects the natural Green/Yellow border better than forcing Hex codes.
    print(f"  - Auto-detecting {NUM_COLORS} colors...")
    quantized = quantize_colors(smooth, NUM_COLORS)

    # 4. MORPHOLOGICAL CLEANING (Scaled Up)
    # In your old script, you used a (3,3) kernel.
    # Since we upscaled 3x, we use a (7,7) kernel to get the same cleaning power,
    # but with much sharper edges on the text.
    print("  - Scrubbing speckles...")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    
    # Remove white specs (Open)
    clean = cv2.morphologyEx(quantized, cv2.MORPH_OPEN, kernel)
    # Remove dark holes (Close)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

    # 5. DOWNSCALE
    # Return to original size. This makes the text look crisp and anti-aliased.
    print("  - Downscaling...")
    final = cv2.resize(clean, (w, h), interpolation=cv2.INTER_AREA)

    out_path = os.path.join(OUTPUT_FOLDER, f"auto_{filename}")
    cv2.imwrite(out_path, final)
    print(f"Saved to {out_path}")

def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for file in files:
        process_image(file)

    print("Batch processing complete.")

if __name__ == "__main__":
    main()