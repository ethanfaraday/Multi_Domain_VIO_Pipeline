import cv2
import numpy as np
import glob
import os

def apply_underwater_effect(image):
    # --- 1. Color shift: add bluish-green tint, reduce red ---
    b, g, r = cv2.split(image.astype(np.float32))
    r *= 0.6      # attenuate red
    g *= 1.0      # keep green
    b *= 1.2      # boost blue
    tinted = cv2.merge([b, g, r])
    tinted = np.clip(tinted, 0, 255).astype(np.uint8)

    # --- 2. Reduce contrast ---
    alpha = 0.7  # contrast < 1
    beta = 20    # brightness shift
    low_contrast = cv2.convertScaleAbs(tinted, alpha=alpha, beta=beta)

    # --- 3. Blur slightly to simulate scattering ---
    blurred = cv2.GaussianBlur(low_contrast, (7, 7), 2)

    # --- 4. Add haze/particles overlay ---
    noise = np.random.normal(0, 15, image.shape).astype(np.int16)
    noisy = cv2.add(blurred.astype(np.int16), noise, dtype=cv2.CV_8U)
    return noisy

# Apply to a sequence of images in a folder
input_folder = r"C:\VIOCODE\ComputerVision\VisualOdometry\curv\left_image"
output_folder = r"C:\\VIOCODE\\ComputerVision\\VisualOdometry\\curv_underwater\\left_image"
os.makedirs(output_folder, exist_ok=True)

for file in glob.glob(os.path.join(input_folder, "*.png")):
    img = cv2.imread(file)
    result = apply_underwater_effect(img)
    out_path = os.path.join(output_folder, os.path.basename(file))
    cv2.imwrite(out_path, result)
