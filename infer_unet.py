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
from queue import Queue

diagnostic_queue = Queue(maxsize=8)

from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from multiprocessing import Pool, cpu_count
import tensorflow as tf
from keras import layers, models
from keras.models import load_model
from backend import xp, GPU_ENABLED
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
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


# Inference configuration
IMG_SIZE = 128
MAX_ASSIGN_FACTOR = 0.6
SEED = 42
PIXEL_PITCH = 5e-6    
FOCAL_LENGTH = 5.2e-3  
VIS_EVERY_N_FRAMES = 2
_vis_counter = 0
_first_vis = True

_A_cached = None
_A_pinv = None
_R_phys_cached = None
_ref_centroids_cached = None
_match_indices_cached = None




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
    fig.patch.set_facecolor("white")

    # -------------------------------
    # Reference IP
    # -------------------------------
    axs[0, 0].imshow(ref_img, cmap="gray", origin="upper")
    axs[0, 0].set_title("Reference (IP)")
    axs[0, 0].axis("off")

    # -------------------------------
    # Aberrated IA
    # -------------------------------
    axs[0, 1].imshow(ab_img, cmap="gray", origin="upper")
    axs[0, 1].set_title("Aberrated (IA)")
    axs[0, 1].axis("off")

    # -------------------------------
    # Dot overlay (CORRECT ORIENTATION + COLOR)
    # -------------------------------
    axs[1, 0].imshow(
        np.zeros_like(ref_img),
        cmap="gray",
        origin="upper"
    )
    axs[1, 0].set_title("IP (green) vs IA (red)")
    axs[1, 0].set_aspect("equal", adjustable="box")
    axs[1, 0].axis("off")

    if len(ref_pts):
        axs[1, 0].scatter(ref_pts[:, 0], ref_pts[:, 1], s=8, c="lime")
    if len(ab_pts):
        axs[1, 0].scatter(ab_pts[:, 0], ab_pts[:, 1], s=4, c="red")

    # -------------------------------
    # Quiver plot (GEOMETRY FIXED)
    # -------------------------------
    axs[1, 1].set_title(title)
    axs[1, 1].axis("off")

    h, w = ref_img.shape
    axs[1, 1].set_xlim(0, w)
    axs[1, 1].set_ylim(0, h)
    axs[1, 1].invert_yaxis()
    axs[1, 1].set_aspect("equal", adjustable="box")

    if len(ref_pts) >= 2 and len(ab_pts) >= 2:
        dists = cdist(ref_pts, ref_pts)
        np.fill_diagonal(dists, np.inf)
        median_spacing = np.median(np.min(dists, axis=1))
        max_dist = MAX_ASSIGN_FACTOR * median_spacing

        cost = cdist(ref_pts, ab_pts)
        r, c = linear_sum_assignment(cost)
        valid = cost[r, c] <= max_dist

        if np.any(valid):
            dx = ab_pts[c[valid], 0] - ref_pts[r[valid], 0]
            dy = ab_pts[c[valid], 1] - ref_pts[r[valid], 1]

            axs[1, 1].quiver(
                ref_pts[r[valid], 0],
                ref_pts[r[valid], 1],
                dx,
                dy,
                angles="xy",
                scale_units="xy",
                scale=1.0,
                color="black",
                width=0.002,
            )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, facecolor="white")
    plt.close(fig)
def save_diagnostic_png(res, output_dir):
    output_dir = Path(output_dir)
    stem = res["name"]

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.patch.set_facecolor("white")
    fig.suptitle(f"Displacement: {stem}", fontsize=12)

    # -------------------------------
    # Reference
    # -------------------------------
    axs[0, 0].imshow(res["ref_img"], cmap="gray", origin="upper")
    axs[0, 0].set_title("Reference (IP)")
    axs[0, 0].axis("off")

    # -------------------------------
    # Aberrated
    # -------------------------------
    axs[0, 1].imshow(res["ab_img"], cmap="gray", origin="upper")
    axs[0, 1].set_title("Aberrated (IA)")
    axs[0, 1].axis("off")

    # -------------------------------
    # Dot overlay (CORRECT)
    # -------------------------------
    axs[1, 0].imshow(
        np.zeros_like(res["ref_img"]),
        cmap="gray",
        origin="upper"
    )
    axs[1, 0].scatter(
        res["ref_centroids"][:, 0],
        res["ref_centroids"][:, 1],
        s=8,
        c="lime",
    )
    axs[1, 0].scatter(
        res["ab_centroids"][:, 0],
        res["ab_centroids"][:, 1],
        s=4,
        c="red",
    )
    axs[1, 0].set_title("IP (green) vs IA (red)")
    axs[1, 0].set_aspect("equal", adjustable="box")
    axs[1, 0].axis("off")

    # -------------------------------
    # Quiver plot (CORRECT SCALE)
    # -------------------------------
    axs[1, 1].set_title("Displacement Field")
    axs[1, 1].axis("off")

    h, w = res["ref_img"].shape
    axs[1, 1].set_xlim(0, w)
    axs[1, 1].set_ylim(0, h)
    axs[1, 1].invert_yaxis()
    axs[1, 1].set_aspect("equal", adjustable="box")

    if len(res["displacements"]) > 0:
        axs[1, 1].quiver(
            res["matched_ref"][:, 0],
            res["matched_ref"][:, 1],
            res["displacements"][:, 0],
            res["displacements"][:, 1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color="black",
            width=0.002,
        )

    fig.savefig(
        output_dir / f"displacement_{stem}.png",
        dpi=90,                     # ↓ big speed gain (this matters most)
        format="png",
        bbox_inches=None,           # 🔴 HUGE speedup vs "tight"
        facecolor="white",
        pil_kwargs={"compress_level": 3},  # fast PNG compression (0–9)
    )



    plt.close(fig)

def infer_pair(ref_image_path, ab_image_path, zernike_modes):
    global _A_cached, _A_pinv, _A_rows

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

    ref_pred = model.predict(ref_resized[np.newaxis, ..., np.newaxis], verbose=0)[0, ..., 0]
    ab_pred  = model.predict(ab_resized[np.newaxis, ..., np.newaxis], verbose=0)[0, ..., 0]

    ref_mask = cleanup_mask((ref_pred > 0.45).astype(np.uint8) * 255)
    ab_mask  = cleanup_mask((ab_pred  > 0.45).astype(np.uint8) * 255)

    ref_mask = cv2.resize(ref_mask, ref_img.shape[::-1], interpolation=cv2.INTER_NEAREST)
    ab_mask  = cv2.resize(ab_mask,  ab_img.shape[::-1],  interpolation=cv2.INTER_NEAREST)

    ref_pts = mask_to_centroids(ref_mask)
    ab_pts  = mask_to_centroids(ab_mask)

    if len(ref_pts) < 50:
        ref_pts = mask_to_centroids(create_binary_mask_from_ip(ref_img))
    if len(ab_pts) < 50:
        ab_pts = mask_to_centroids(create_binary_mask_from_ip(ab_img))

    matched_ref = np.empty((0, 2))
    matched_ab  = np.empty((0, 2))
    displacements_px = np.empty((0, 2))

    if len(ref_pts) >= 2 and len(ab_pts) >= 2:
        dists = cdist(ref_pts, ref_pts)
        np.fill_diagonal(dists, np.inf)
        max_dist = MAX_ASSIGN_FACTOR * np.median(np.min(dists, axis=1))

        cost = cdist(ref_pts, ab_pts)
        r, c = linear_sum_assignment(cost)
        valid = cost[r, c] <= max_dist

        matched_ref = ref_pts[r[valid]]
        matched_ab  = ab_pts[c[valid]]
        displacements_px = matched_ab - matched_ref

    zernike_coeffs = None
    wavefront = None
    wavefront_vis = None
    X = Y = None

    if len(displacements_px) >= len(zernike_modes):
        x_norm, y_norm, R_px = normalize_centroids(matched_ref)
        R_phys = R_px * PIXEL_PITCH

        b = -build_slope_vector(displacements_px * PIXEL_PITCH) / FOCAL_LENGTH

        if _A_cached is not None and b.shape[0] == _A_rows:
            zernike_coeffs = _A_pinv @ b
        else:
            A = build_design_matrix(x_norm, y_norm, zernike_modes, R_phys)
            zernike_coeffs = solve_modal_coefficients(A, b)

            if _A_cached is None:
                _A_cached = A
                _A_pinv = np.linalg.pinv(A)
                _A_rows = A.shape[0]
                print("📌 Design matrix built and cached (safe)")

        X, Y, W_phys = reconstruct_wavefront(zernike_coeffs, zernike_modes)

        from backend import to_cpu
        X, Y, W_phys, zernike_coeffs = map(to_cpu, (X, Y, W_phys, zernike_coeffs))

        if np.isfinite(W_phys).any():
            wavefront = W_phys
            W_vis = (W_phys - np.nanmean(W_phys)) / 80e-9
            W_vis[(X**2 + Y**2) > 1.0] = 0.0
            wavefront_vis = np.flipud(W_vis).T

    return {
        "name": Path(ref_image_path).stem,
        "ref_img": ref_img,
        "ab_img": ab_img,
        "ref_centroids": ref_pts,
        "ab_centroids": ab_pts,
        "matched_ref": matched_ref,
        "matched_ab": matched_ab,
        "displacements": displacements_px,
        "zernike_coeffs": zernike_coeffs,
        "wavefront": wavefront,
        "wavefront_vis": wavefront_vis,
        "X": X,
        "Y": Y,
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

def save_inference_outputs(res, output_dir, frame_index):
    """
    Centralized saving for inference outputs.
    """
    output_dir = Path(output_dir)

    # Wavefront plot
    if "wavefront_fig" in res:
        res["wavefront_fig"].savefig(
            output_dir / f"wavefront_{frame_index:04d}.png",
            dpi=150,
            bbox_inches="tight",
        )

    # Displacement plot
    if "displacement_fig" in res:
        res["displacement_fig"].savefig(
            output_dir / f"displacement_{frame_index:04d}.png",
            dpi=150,
            bbox_inches="tight",
        )

    # Optional raw arrays
    if "wavefront" in res:
        np.save(output_dir / f"wavefront_{frame_index:04d}.npy", res["wavefront"])
def discover_pairs(ref_input, ab_input):
    """Discover matching (IP, IA) frame pairs."""
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

    return frame_pairs
def infer_manager(
    ref_input,
    ab_input,
    save_outputs=True,
    output_root=OUTPUT_DIR,
    n_zernike=10,
    emit_result=None,
):
    zernike_modes = generate_zernike_modes(n_zernike)

    output_dir = make_timestamped_output_dir(output_root) if save_outputs else None

    frame_pairs = discover_pairs(ref_input, ab_input)
    print(f"▶ Processing {len(frame_pairs)} frame pairs (serial)")

    for idx, (ref_path, ab_path) in enumerate(frame_pairs):
        t0 = time.perf_counter()

        res = infer_pair(ref_path, ab_path, zernike_modes)
        if res is None:
            continue

        # 🔴 LIVE UPDATE — MUST COME FIRST
        if emit_result is not None:
            emit_result(res)

        if save_outputs:
            try:
                diagnostic_queue.put_nowait((res, output_dir))
            except Exception:
                pass  # queue full → drop diagnostic

        print(f"[Frame {idx:04d}] Compute: {(time.perf_counter() - t0)*1000:.2f} ms")

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
