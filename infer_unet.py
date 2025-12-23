import os

# Base directory (location of infer_unet.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset root (optional default, not mandatory for inference)

# Trained model path (must exist)
MODEL_PATH = os.path.join(BASE_DIR, "unet_trained.keras")

# Output directory for saved results
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
import cv2
import time
from PyQt6.QtCore import QEventLoop
from PyQt6 import QtWidgets
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from multiprocessing import Pool, cpu_count
import tensorflow as tf
from keras import layers, models
from keras.models import load_model
from wavefront_modal import (
    normalize_centroids,
    build_design_matrix,
    build_slope_vector,
    remove_piston_tilt,
    solve_modal_coefficients,
    reconstruct_wavefront,
    plot_wavefront,
    generate_zernike_modes
)
from wavefront_live import LiveWavefrontViewer
wavefront_viewer = LiveWavefrontViewer(grid_size=200)


# Inference configuration
IMG_SIZE = 128
MAX_ASSIGN_FACTOR = 0.6
SEED = 42
# physical acquisition parameters (set these to your sensor/lens)
PIXEL_PITCH = 5e-6    # meters per pixel (example; change to your camera)
FOCAL_LENGTH = 5.2e-3    # focal length in meters (example)
VIS_EVERY_N_FRAMES = 2
_vis_counter = 0
_first_vis = True



# Reproducibility
tf.random.set_seed(SEED)
np.random.seed(SEED)
def _infer_worker(args):
    ref_path, ab_path, output_dir, zernike_modes = args
    return infer_pair(
        ref_path,
        ab_path,
        output_dir=output_dir,
        zernike_modes=zernike_modes
    )

def make_timestamped_output_dir(base_dir):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(base_dir, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def radial_compensate(img):
    h, w = img.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((X - cx)**2 + (Y - cy)**2)
    r_norm = r / r.max()

    nbins = 80
    bins = np.linspace(0.0, 1.0, nbins + 1)
    median_profile = np.zeros(nbins)
    for i in range(nbins):
        mask = (r_norm >= bins[i]) & (r_norm < bins[i+1])
        median_profile[i] = np.median(img[mask]) if np.any(mask) else median_profile[i-1] if i > 0 else 1.0
    median_profile = cv2.blur(median_profile.reshape(-1,1), (5,1)).ravel()
    median_profile = np.maximum(median_profile, 1e-4)

    gain_profile = median_profile.max() / median_profile
    gain_image = np.interp(r_norm.ravel(), 0.5*(bins[:-1]+bins[1:]), gain_profile).reshape(h,w)
    gain_image = np.clip(gain_image, 1.0, 4.5)

    compensated = np.clip(img * gain_image, 0.0, 1.0)
    return compensated, gain_image
def create_binary_mask_from_ip(ip_img, debug_show=False):
    imgf = ip_img.astype(np.float32) / 255.0
    comp, gain = radial_compensate(imgf)
    blur = cv2.GaussianBlur((comp*255).astype(np.uint8), (5,5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(th, connectivity=8)
    out = np.zeros_like(th)
    min_area = 2
    for lab in range(1, num_labels):
        if stats[lab, cv2.CC_STAT_AREA] >= min_area:
            out[labels == lab] = 255

    if debug_show:
        fig, ax = plt.subplots(1,3, figsize=(12,4))
        ax[0].imshow(imgf, cmap='gray'); ax[0].set_title('orig'); ax[0].axis('off')
        ax[1].imshow(comp, cmap='gray'); ax[1].set_title('compensated'); ax[1].axis('off')
        ax[2].imshow(out, cmap='gray'); ax[2].set_title('binary mask'); ax[2].axis('off')
        plt.show()
    return out

def load_and_prepare_image(path):
    """Load a single grayscale image and resize for U-Net inference."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")

    resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return img, normalized[np.newaxis, ..., np.newaxis]

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"❌ Trained model not found at {MODEL_PATH}. "
        "Please run train_unet.py first."
    )

print(f"📦 Loading trained model from {MODEL_PATH}")
model = load_model(MODEL_PATH)

def mask_to_centroids(mask):
    """Extracts centroids from a binary mask."""
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    pts = []
    for i in range(1, len(centroids)):  # skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 3:
            pts.append(centroids[i])
    return np.array(pts) if pts else np.zeros((0, 2), dtype=float)

def cleanup_mask(mask):
    """Remove tiny components."""
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    for i in range(1, len(stats)):
        if stats[i, cv2.CC_STAT_AREA] >= 3:
            clean[labels == i] = 255
    return clean

def match_and_visualize(ref_pts, ab_pts, ref_img, ab_img, title, save_path=None):
    """Visualize matches: 4 views — IP, IA, dot overlay, and quiver displacement."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Subplot 1: Reference IP
    axs[0, 0].imshow(ref_img, cmap='gray')
    axs[0, 0].set_title('Reference (IP)')
    axs[0, 0].axis('off')

    # Subplot 2: Aberrated IA
    axs[0, 1].imshow(ab_img, cmap='gray')
    axs[0, 1].set_title('Aberrated (IA)')
    axs[0, 1].axis('off')

    # Subplot 3: Colored dot overlay (black bg)
    dot_canvas = np.zeros_like(ref_img)
    axs[1, 0].imshow(dot_canvas, cmap='gray')
    axs[1, 0].set_title('IP (green) vs IA (red)')
    axs[1, 0].axis('off')
    for (x, y) in ref_pts:
        axs[1, 0].plot(x, y, 'go', markersize=3)
    for (x, y) in ab_pts:
        axs[1, 0].plot(x, y, 'ro', markersize=3)

    # Subplot 4: Quiver plot
    axs[1, 1].set_title(title)
    axs[1, 1].set_facecolor('white')
    axs[1, 1].axis('off')
    h, w = ref_img.shape
    axs[1, 1].set_xlim(0, w)
    axs[1, 1].set_ylim(h, 0)  # invert Y for image coordinates

    # Match points
    if len(ref_pts) < 2 or len(ab_pts) < 2:
        axs[1, 1].text(0.5, 0.5, "Too few points to match", ha='center', va='center', transform=axs[1, 1].transAxes)
        print("Too few centroids to attempt matching.")
    else:
        # Compute max match distance
        dists = cdist(ref_pts, ref_pts)
        np.fill_diagonal(dists, np.inf)
        median_spacing = np.median(np.min(dists, axis=1))
        max_dist = MAX_ASSIGN_FACTOR * median_spacing

        # Hungarian assignment
        cost = cdist(ref_pts, ab_pts)
        row_ind, col_ind = linear_sum_assignment(cost)
        matches = [(r, c) for r, c in zip(row_ind, col_ind) if cost[r, c] <= max_dist]

        print(f"Relax: factor={MAX_ASSIGN_FACTOR:.1f}, thresh={max_dist:.3f}, matches={len(matches)}")

        if matches:
            for r, c in matches:
                x0, y0 = ref_pts[r]
                x1, y1 = ab_pts[c]
                dx, dy = x1 - x0, y1 - y0
                axs[1, 1].arrow(x0, y0, dx, dy, head_width=0.0, head_length=0.0,
                                color='black', linewidth=0.6)
        else:
            axs[1, 1].text(0.5, 0.5, "No matches found", ha='center', va='center', transform=axs[1, 1].transAxes)
            print("No matches found even after relaxation – inspect centroids above.")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)

    # Do NOT block execution in inference
    plt.close(fig)
def infer_pair(ref_image_path, ab_image_path, output_dir, zernike_modes):
    ref_img = cv2.imread(str(ref_image_path), cv2.IMREAD_GRAYSCALE)
    ab_img  = cv2.imread(str(ab_image_path), cv2.IMREAD_GRAYSCALE)

    if ref_img is None or ab_img is None:
        print(f"❌ Could not read images: {ref_image_path}")
        return None

    # ----------------------------------
    # Resize & normalize for U-Net
    # ----------------------------------
    ref_resized = cv2.resize(ref_img, (IMG_SIZE, IMG_SIZE)) / 255.0
    ab_resized  = cv2.resize(ab_img,  (IMG_SIZE, IMG_SIZE)) / 255.0

    # ----------------------------------
    # Predict masks
    # ----------------------------------
    ref_pred = model.predict(
        ref_resized[np.newaxis, ..., np.newaxis],
        verbose=0
    )[0, ..., 0]

    ab_pred = model.predict(
        ab_resized[np.newaxis, ..., np.newaxis],
        verbose=0
    )[0, ..., 0]

    ref_mask = (ref_pred > 0.45).astype(np.uint8) * 255
    ab_mask  = (ab_pred  > 0.45).astype(np.uint8) * 255

    ref_mask = cv2.resize(ref_mask, ref_img.shape[::-1], interpolation=cv2.INTER_NEAREST)
    ab_mask  = cv2.resize(ab_mask,  ab_img.shape[::-1],  interpolation=cv2.INTER_NEAREST)

    ref_mask = cleanup_mask(ref_mask)
    ab_mask  = cleanup_mask(ab_mask)

    # ----------------------------------
    # Extract centroids
    # ----------------------------------
    ref_pts = mask_to_centroids(ref_mask)
    ab_pts  = mask_to_centroids(ab_mask)

    if len(ab_pts) < 50:
        ab_pts = mask_to_centroids(create_binary_mask_from_ip(ab_img))
    if len(ref_pts) < 50:
        ref_pts = mask_to_centroids(create_binary_mask_from_ip(ref_img))

    # ----------------------------------
    # Hungarian matching
    # ----------------------------------
    matched_ref = np.empty((0, 2))
    matched_ab  = np.empty((0, 2))
    displacements_px = np.empty((0, 2))

    if len(ref_pts) >= 2 and len(ab_pts) >= 2:
        dists = cdist(ref_pts, ref_pts)
        np.fill_diagonal(dists, np.inf)
        median_spacing = np.median(np.min(dists, axis=1))
        max_dist = MAX_ASSIGN_FACTOR * median_spacing

        cost = cdist(ref_pts, ab_pts)
        row_ind, col_ind = linear_sum_assignment(cost)

        valid = cost[row_ind, col_ind] <= max_dist
        if np.any(valid):
            matched_ref = ref_pts[row_ind[valid]]
            matched_ab  = ab_pts[col_ind[valid]]
            displacements_px = matched_ab - matched_ref

    # ----------------------------------
    # Save displacement visualization
    # ----------------------------------
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        out_name = f"displacement_{Path(ref_image_path).stem}.png"
        output_path = os.path.join(output_dir, out_name)

        match_and_visualize(
            ref_pts,
            ab_pts,
            ref_img,
            ab_img,
            title=f"Displacement: {Path(ref_image_path).name}",
            save_path=output_path
        )

    # ----------------------------------
    # Modal reconstruction
    # ----------------------------------
    zernike_coeffs = None
    wavefront = None
    wavefront_vis = None
    X = Y = None

    if len(displacements_px) >= len(zernike_modes):
        x_norm, y_norm, R_px = normalize_centroids(matched_ref)
        R_phys = R_px * PIXEL_PITCH

        displacements_m = displacements_px * PIXEL_PITCH
        b = -build_slope_vector(displacements_m) / FOCAL_LENGTH

        A = build_design_matrix(x_norm, y_norm, zernike_modes, R_phys)
        zernike_coeffs = solve_modal_coefficients(A, b)

        X, Y, W_phys = reconstruct_wavefront(zernike_coeffs, zernike_modes)

        if W_phys is not None and np.isfinite(W_phys).any():
            wavefront = W_phys

            # -------------------------------
            # Visualization-only conditioning
            # -------------------------------
            W_vis = W_phys - np.nanmean(W_phys)
            Z_VIS_SCALE = 80e-9
            W_vis /= Z_VIS_SCALE

            pupil_mask = (X**2 + Y**2) <= 1.0
            W_vis[~pupil_mask] = 0.0

            W_vis = np.flipud(W_vis).T
            wavefront_vis = W_vis

    # ----------------------------------
    # RETURN PURE DATA (NO UI)
    # ----------------------------------
    return {
        "name": Path(ref_image_path).name,
        "ref_centroids": ref_pts,
        "ab_centroids": ab_pts,
        "displacements": displacements_px,
        "num_matches": len(displacements_px),
        "zernike_coeffs": zernike_coeffs,

        # wavefronts
        "wavefront": wavefront,
        "wavefront_vis": wavefront_vis,
        "X": X,
        "Y": Y,

        "output_path": output_dir
    }


def infer_folder(ref_folder, ab_folder, output_dir):
    ref_files = sorted(
        p for p in Path(ref_folder).iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    )

    if not ref_files:
        print(f"❌ No image files found in {ref_folder}")
        return

    for ref_path in ref_files:
        # Expect pattern like IP5_00001.png or IP_00001.png
        stem = ref_path.stem

        if "_" not in stem:
            print(f"⚠️ Skipping unrecognized filename: {ref_path.name}")
            continue

        # Split only on last underscore to preserve prefix
        prefix, index = stem.rsplit("_", 1)

        if not index.isdigit():
            print(f"⚠️ Invalid index in filename: {ref_path.name}")
            continue

        # Replace IP → IA only once (safe)
        ab_prefix = prefix.replace("IP", "IA", 1)
        ab_name = f"{ab_prefix}_{index}{ref_path.suffix}"
        ab_path = Path(ab_folder) / ab_name

        if not ab_path.exists():
            print(f"⚠️ Missing aberrated file: {ab_name}")
            continue

        infer_pair(ref_path, ab_path, output_dir)

def infer_manager(
    ref_input,
    ab_input,
    save_outputs=True,
    output_root=OUTPUT_DIR,
    n_zernike=10,
):
    zernike_modes = generate_zernike_modes(n_zernike)

    # -------------------------------
    # Output directory
    # -------------------------------
    if save_outputs:
        if output_root is None:
            raise ValueError("output_root must be provided when save_outputs=True")
        output_dir = make_timestamped_output_dir(output_root)
    else:
        output_dir = None

    results = []

    # -------------------------------
    # Build frame pairs
    # -------------------------------
    frame_pairs = []

    if os.path.isdir(ref_input) and os.path.isdir(ab_input):
        ref_files = sorted(
            p for p in Path(ref_input).iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        )

        for ref_path in ref_files:
            stem = ref_path.stem
            if "_" not in stem:
                continue

            prefix, index = stem.rsplit("_", 1)
            if not index.isdigit():
                continue

            ab_prefix = prefix.replace("IP", "IA", 1)
            ab_name = f"{ab_prefix}_{index}{ref_path.suffix}"
            ab_path = Path(ab_input) / ab_name

            if ab_path.exists():
                frame_pairs.append((ref_path, ab_path))

    elif os.path.isfile(ref_input) and os.path.isfile(ab_input):
        frame_pairs.append((Path(ref_input), Path(ab_input)))

    else:
        raise ValueError("ref_input and ab_input must both be files or folders")

    if not frame_pairs:
        print("⚠️ No valid frame pairs found.")
        return []

    print(f"▶ Processing {len(frame_pairs)} frame pairs (serial)")

    # -------------------------------
    # SERIAL execution (CRITICAL)
    # -------------------------------
    for idx, (ref_path, ab_path) in enumerate(frame_pairs):
        t0 = time.perf_counter()

        # ---- FULL COMPUTATION ----
        res = infer_pair(
            ref_path,
            ab_path,
            output_dir=output_dir,
            zernike_modes=zernike_modes,
        )
        t1 = time.perf_counter()

        if res is None:
            continue

        results.append(res)

        # ---- LIVE VISUALIZATION ----
        if res.get("wavefront") is not None:
            wavefront_viewer.update(
                W_vis=res["wavefront_vis"],
                W_phys=res["wavefront"],
                X=res["X"],
                Y=res["Y"],
            )
        t2 = time.perf_counter()

        # ---- PRINT EVERY N FRAMES ----
        if idx % 5 == 0:
            print(
                f"[Frame {idx:04d}] "
                f"Compute: {(t1 - t0)*1000:.2f} ms | "
                f"Render: {(t2 - t1)*1000:.2f} ms | "
                f"Total: {(t2 - t0)*1000:.2f} ms"
            )
    return results
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="U-Net inference for dot displacement")
    parser.add_argument("--ref", required=True)
    parser.add_argument("--ab", required=True)
    parser.add_argument(
    "--out",
    default=None,
    help="Optional output directory (timestamped folder used if omitted)"
    )

    args = parser.parse_args()
    # Decide output directory
    if args.out is None:
        output_dir = make_timestamped_output_dir(OUTPUT_DIR)
        print(f"📁 Auto output directory: {output_dir}")
    else:
        output_dir = args.out
        os.makedirs(output_dir, exist_ok=True)

    print("⏱️ Starting inference...")
    start_time = time.time()

    # Folder mode
    if os.path.isdir(args.ref) and os.path.isdir(args.ab):
        infer_folder(
            ref_folder=args.ref,
            ab_folder=args.ab,
            output_dir=output_dir
        )
    # Single-image mode
    elif os.path.isfile(args.ref) and os.path.isfile(args.ab):
        infer_pair(
            ref_image_path=args.ref,
            ab_image_path=args.ab,
            output_dir=output_dir
        )
    else:
        raise ValueError("ref and ab must both be folders or both be files")

    end_time = time.time()
    elapsed = end_time - start_time

    print(f"✅ Inference completed in {elapsed:.2f} seconds")
