# aerial.py
# Create a synthetic "aerial" version of an image sequence by applying:
# 1) altitude/resolution loss (downscale+upscale),
# 2) haze/airlight,
# 3) linear motion blur,
# 4) mild desaturation and exposure shift,
# 5) optional rolling-shutter skew,
# 6) optional sun glare overlay.
#
# Paths mirror your underwater.py style. Edit INPUT/OUTPUT paths as needed.

import cv2
import numpy as np
import glob
import os

# --------------------------
# Tunable parameters
# --------------------------
SEED = 1234               # set to None for non-deterministic effects
SCALE = 0.5               # 0.3–0.7 typical. Smaller => stronger altitude effect
HAZE_STRENGTH = 0.35      # 0 (none) .. ~0.6 (very hazy)
MOTION_BLUR_K = 15        # odd integer; length of linear motion blur
DESAT_FACTOR = 0.88       # 0.7–1.0; lower => more desaturated
BRIGHT_GAIN = 1.08        # 1.0–1.2; small lift to mimic AE in bright scenes

ENABLE_ROLLING_SHUTTER = False  # set True to add vertical shear per scanline
RS_MAX_SHEAR = 0.03             # ~0.01–0.05 reasonable if enabled

ENABLE_SUN_GLARE = False        # set True to add faint sun-spot glare
GLARE_ALPHA = 0.12              # 0.05–0.2; opacity of glare
GLARE_RADIUS_FRAC = 0.35        # spot radius as fraction of min(h, w)

# --------------------------
# I/O folders (edit to taste)
# --------------------------
# You can run separately for left/right, or point to any folder of frames.
INPUT_FOLDER  = r"C:\VIOCODE\ComputerVision\VisualOdometry\squigle\left_image"
OUTPUT_FOLDER = r"C:\VIOCODE\ComputerVision\VisualOdometry\squigle_aerial\left_image"

# If you want to batch both L/R in one go, list them here:
PAIRS = [
    # straight_line
    (r"C:\VIOCODE\ComputerVision\VisualOdometry\figure_8_run_2\left_image",
     r"C:\VIOCODE\ComputerVision\VisualOdometry\figure_8_run_2_aerial\left_image"),
    (r"C:\VIOCODE\ComputerVision\VisualOdometry\figure_8_run_2\right_image",
     r"C:\VIOCODE\ComputerVision\VisualOdometry\figure_8_run_2_aerial\right_image"),
]


def _apply_motion_blur(img, k):
    """Apply horizontal linear motion blur with kernel size k (odd)."""
    k = max(3, int(k) | 1)  # ensure odd and >=3
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0
    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel)


def _apply_haze(img, strength=0.35, airlight=255):
    """
    Simple depth-independent haze/airlight: I' = I*(1-s) + A*s
    strength in [0..1], A ~ 255 (bright sky).
    """
    fog = np.full_like(img, fill_value=airlight, dtype=np.uint8)
    # blend without clipping
    return cv2.addWeighted(img, 1.0 - strength, fog, strength, 0.0)


def _downscale_then_upscale(img, scale=0.5):
    """Simulate altitude: reduce texture/resolution but keep original size."""
    h, w = img.shape[:2]
    w2, h2 = max(1, int(w * scale)), max(1, int(h * scale))
    small = cv2.resize(img, (w2, h2), interpolation=cv2.INTER_AREA)
    back = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return back


def _apply_desat_and_brightness(img, desat=0.9, gain=1.05):
    """Reduce saturation slightly and lift brightness a touch."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    # S-channel
    hsv[..., 1] *= float(desat)
    # V-channel
    hsv[..., 2] *= float(gain)
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _apply_rolling_shutter_skew(img, max_shear=0.02, rng=None):
    """
    Row-wise shear to mimic rolling-shutter under lateral motion.
    Uses a gentle sinusoidal shear profile from top->bottom.
    """
    if rng is None:
        rng = np.random.default_rng()
    h, w = img.shape[:2]
    shear = float(rng.uniform(-max_shear, max_shear))
    # Build per-row affine transforms
    out = np.empty_like(img)
    # Sinusoidal modulation so skew varies across the frame
    y_coords = np.linspace(0, np.pi, h, dtype=np.float32)
    offsets = np.sin(y_coords) * shear * w  # pixel offset per row

    for y in range(h):
        M = np.float32([[1, 0, offsets[y]], [0, 1, 0]])
        out[y:y+1, :, :] = cv2.warpAffine(img[y:y+1, :, :], M, (w, 1), flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_REPLICATE)
    return out


def _apply_sun_glare(img, alpha=0.12, radius_frac=0.35, rng=None):
    """Overlay a soft circular glare spot (simulating sun) at a random edge location."""
    if rng is None:
        rng = np.random.default_rng()
    h, w = img.shape[:2]
    r = int(min(h, w) * float(radius_frac))

    # pick a corner-ish center
    centers = [(int(0.15*w), int(0.2*h)),
               (int(0.85*w), int(0.2*h)),
               (int(0.15*w), int(0.15*h)),
               (int(0.85*w), int(0.15*h))]
    cx, cy = centers[int(rng.integers(0, len(centers)))]

    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    mask = np.clip(1.0 - (dist / float(r)), 0.0, 1.0)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=r/3.0, sigmaY=r/3.0)

    # glare color slightly warm
    glare = np.zeros_like(img, dtype=np.float32)
    glare[..., 0] = 255.0 * mask * 0.9  # B
    glare[..., 1] = 255.0 * mask * 1.0  # G
    glare[..., 2] = 255.0 * mask * 1.1  # R

    blend = cv2.addWeighted(img.astype(np.float32), 1.0, glare, float(alpha), 0.0)
    return np.clip(blend, 0, 255).astype(np.uint8)


def apply_aerial_effect(image, rng=None):
    """Full aerial pipeline applied in-place to a BGR uint8 image."""
    if rng is None:
        rng = np.random.default_rng()

    img = image.astype(np.uint8, copy=False)

    # 1) Altitude/resolution loss while preserving original dimensions
    img = _downscale_then_upscale(img, scale=SCALE)

    # 2) Haze / airlight
    img = _apply_haze(img, strength=HAZE_STRENGTH, airlight=255)

    # 3) Linear motion blur (horizontal)
    img = _apply_motion_blur(img, k=MOTION_BLUR_K)

    # 4) Mild desaturation and brightness lift
    img = _apply_desat_and_brightness(img, desat=DESAT_FACTOR, gain=BRIGHT_GAIN)

    # 5) Optional rolling-shutter skew
    if ENABLE_ROLLING_SHUTTER:
        img = _apply_rolling_shutter_skew(img, max_shear=RS_MAX_SHEAR, rng=rng)

    # 6) Optional sun glare overlay
    if ENABLE_SUN_GLARE:
        img = _apply_sun_glare(img, alpha=GLARE_ALPHA, radius_frac=GLARE_RADIUS_FRAC, rng=rng)

    return img


def _process_folder(input_folder, output_folder, seed=SEED):
    if seed is not None:
        np.random.seed(seed)
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    os.makedirs(output_folder, exist_ok=True)

    patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(input_folder, pat)))
    files.sort()

    if not files:
        print(f"[WARN] No images found in: {input_folder}")
        return

    print(f"[INFO] Processing {len(files)} images from:\n  {input_folder}\n-> {output_folder}")
    for idx, f in enumerate(files, 1):
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Skipping unreadable file: {f}")
            continue
        result = apply_aerial_effect(img, rng=rng)
        out_path = os.path.join(output_folder, os.path.basename(f))
        ok = cv2.imwrite(out_path, result)
        if not ok:
            print(f"[ERR ] Failed to write: {out_path}")
        if idx % 50 == 0 or idx == len(files):
            print(f"[INFO] {idx}/{len(files)} done")


if __name__ == "__main__":
    # Option A: single folder (mirrors your underwater.py)
    # _process_folder(INPUT_FOLDER, OUTPUT_FOLDER)

    # Option B: batch L/R folders in one go — uncomment and edit PAIRS above.
    for inp, out in PAIRS:
        _process_folder(inp, out)
