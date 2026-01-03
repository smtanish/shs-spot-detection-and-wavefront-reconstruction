# This file runs the full Shack–Hartmann inference pipeline using a trained U-Net model.
# It detects spot locations in reference (IP) and aberrated (IA) images, matches them using a global assignment method, converts pixel shifts into physical slopes, and reconstructs the wavefront using a Zernike modal fit.
# The pipeline supports single images or folders,  live visualization, saving of diagnostics, and caching for real-time performance.


import os
# Get the folder where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Path to the trained U-Net model used for inference
MODEL_PATH = os.path.join(BASE_DIR, "unet_trained.keras")
# Folder where inference outputs and diagnostics will be saved
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
# Create the output folder if it does not already exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
import cv2
import time
from queue import Queue
# Queue used to save diagnostic images without blocking inference
diagnostic_queue = Queue(maxsize=8)
from datetime import datetime
import numpy as np
import matplotlib
# Use a non-interactive backend so plots can be saved without a display
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from multiprocessing import Pool, cpu_count
import tensorflow as tf
from keras import layers, models
from keras.models import load_model
# xp is a NumPy-like interface that allows CPU or GPU use transparently. This is not being used in the current version but may be useful for future optimizations.
from backend import xp, GPU_ENABLED
# Import wavefront reconstruction utilities
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
# Enable dynamic GPU memory allocation to avoid reserving all memory
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
# Size to which images are resized before U-Net inference
IMG_SIZE = 128
# Factor that limits how far a spot is allowed to move during matching
MAX_ASSIGN_FACTOR = 0.6
# Fixed seed to keep inference behavior consistent
SEED = 42
# Physical size of one camera pixel (meters per pixel)
PIXEL_PITCH = 5e-6
# Focal length of the lenslet array (meters)
FOCAL_LENGTH = 5.2e-3
# Control how often visualization is updated during live inference
VIS_EVERY_N_FRAMES = 2
_vis_counter = 0
_first_vis = True
# Cached variables to avoid rebuilding matrices every frame
_A_cached = None
_A_pinv = None
_R_phys_cached = None
_ref_centroids_cached = None
_match_indices_cached = None

# Set fixed seeds so results stay consistent across runs
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Helper function used for parallel inference on image pairs
def _infer_worker(args):
    # Unpack inputs for a single inference call
    ref_path, ab_path, output_dir, zernike_modes = args
    # Run inference on one reference–aberrated image pair
    return infer_pair(
        ref_path,
        ab_path,
        output_dir=output_dir,
        zernike_modes=zernike_modes
    )

# Create a new output folder using the current date and time
def make_timestamped_output_dir(base_dir):
    # Generate a readable timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Build full output path
    out_dir = os.path.join(base_dir, timestamp)
    # Ensure the directory exists
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

# This function compensates for brightness drop from center to edge of the image
def radial_compensate(img):
    # Get image height and width
    h, w = img.shape
    # Compute image center
    cy, cx = h // 2, w // 2
    # Create coordinate grids
    Y, X = np.ogrid[:h, :w]
    # Compute distance of each pixel from the center
    r = np.sqrt((X - cx)**2 + (Y - cy)**2)
    # Normalize distance so values range from 0 to 1
    r_norm = r / r.max()
    # Number of radial bins used to estimate brightness profile
    nbins = 80
    bins = np.linspace(0.0, 1.0, nbins + 1)
    # Store median brightness for each radial ring
    median_profile = np.zeros(nbins)
    for i in range(nbins):
        # Select pixels inside the current radial ring
        mask = (r_norm >= bins[i]) & (r_norm < bins[i+1])
        # Use median to reduce sensitivity to noise
        median_profile[i] = np.median(img[mask]) if np.any(mask) else median_profile[i-1] if i > 0 else 1.0
    # Smooth the radial profile to avoid sharp gain changes
    median_profile = cv2.blur(median_profile.reshape(-1,1), (5,1)).ravel()
    # Avoid division by very small values
    median_profile = np.maximum(median_profile, 1e-4)
    # Compute gain needed to flatten brightness across the image
    gain_profile = median_profile.max() / median_profile
    # Expand the radial gain into a full image-sized gain map
    gain_image = np.interp(r_norm.ravel(), 0.5*(bins[:-1]+bins[1:]), gain_profile).reshape(h,w)
    # Limit gain to prevent excessive noise amplification
    gain_image = np.clip(gain_image, 1.0, 4.5)
    # Apply gain and keep pixel values within valid range
    compensated = np.clip(img * gain_image, 0.0, 1.0)
    return compensated, gain_image

# This function creates a clean binary mask of bright spots from an input image
def create_binary_mask_from_ip(ip_img, debug_show=False):
    # Convert image to float and normalize values to 0–1
    imgf = ip_img.astype(np.float32) / 255.0
    # Compensate for brightness drop toward the edges
    comp, gain = radial_compensate(imgf)
    # Blur slightly to reduce noise and smooth spot shapes
    blur = cv2.GaussianBlur((comp*255).astype(np.uint8), (5,5), 0)
    # Automatically threshold the image to separate spots from background
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Create a small circular kernel for cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    # Fill small gaps inside spots
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    # Remove small isolated noise blobs
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    # Label all connected bright regions
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(th, connectivity=8)
    # Create an empty output mask
    out = np.zeros_like(th)
    # Minimum area required for a region to be considered a real spot
    min_area = 2
    for lab in range(1, num_labels):
        # Keep only regions large enough to be valid spots
        if stats[lab, cv2.CC_STAT_AREA] >= min_area:
            out[labels == lab] = 255
    # Optional visualization for debugging the mask generation
    if debug_show:
        fig, ax = plt.subplots(1,3, figsize=(12,4))
        ax[0].imshow(imgf, cmap='gray'); ax[0].set_title('orig'); ax[0].axis('off')
        ax[1].imshow(comp, cmap='gray'); ax[1].set_title('compensated'); ax[1].axis('off')
        ax[2].imshow(out, cmap='gray'); ax[2].set_title('binary mask'); ax[2].axis('off')
        plt.show()
    return out

# This function loads one image and prepares it for U-Net inference
def load_and_prepare_image(path):
    # Read the image in grayscale
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # Stop if the image cannot be loaded
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    # Resize image to match U-Net input size
    resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    # Normalize pixel values to 0–1
    normalized = resized.astype(np.float32) / 255.0
    # Return original image and the model-ready version
    return img, normalized[np.newaxis, ..., np.newaxis]

# Ensure a trained model exists before running inference
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"❌ Trained model not found at {MODEL_PATH}. "
        "Please run train_unet.py first."
    )
# Load the trained U-Net model
print(f"📦 Loading trained model from {MODEL_PATH}")
model = load_model(MODEL_PATH)

# This function extracts spot center positions from a binary mask
def mask_to_centroids(mask):
    # Label all connected regions in the mask
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    pts = []
    # Skip label 0 because it represents the background
    for i in range(1, len(centroids)):
        # Ignore very small regions that are likely noise
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 3:
            pts.append(centroids[i])
    # Return an empty array if no valid spots are found
    return np.array(pts) if pts else np.zeros((0, 2), dtype=float)

# This function removes very small blobs from a binary mask
def cleanup_mask(mask):
    # Label all connected regions in the mask
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    for i in range(1, len(stats)):
        # Keep only regions large enough to be meaningful
        if stats[i, cv2.CC_STAT_AREA] >= 3:
            clean[labels == i] = 255
    return clean

# This function visualizes spot matching and displacement between reference and aberrated images
def match_and_visualize(ref_pts, ab_pts, ref_img, ab_img, title, save_path=None):
    # Create a 2x2 figure for visualization
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.patch.set_facecolor("white")
    # Show the reference input-plane image
    axs[0, 0].imshow(ref_img, cmap="gray", origin="upper")
    axs[0, 0].set_title("Reference (IP)")
    axs[0, 0].axis("off")
    # Show the aberrated image
    axs[0, 1].imshow(ab_img, cmap="gray", origin="upper")
    axs[0, 1].set_title("Aberrated (IA)")
    axs[0, 1].axis("off")
    # Overlay detected spot positions for reference and aberrated images
    axs[1, 0].imshow(np.zeros_like(ref_img), cmap="gray", origin="upper")
    axs[1, 0].set_title("IP (green) vs IA (red)")
    axs[1, 0].set_aspect("equal", adjustable="box")
    axs[1, 0].axis("off")
    if len(ref_pts):
        axs[1, 0].scatter(ref_pts[:, 0], ref_pts[:, 1], s=8, c="lime")
    if len(ab_pts):
        axs[1, 0].scatter(ab_pts[:, 0], ab_pts[:, 1], s=4, c="red")
    # Prepare the displacement (arrow) plot
    axs[1, 1].set_title(title)
    axs[1, 1].axis("off")
    h, w = ref_img.shape
    axs[1, 1].set_xlim(0, w)
    axs[1, 1].set_ylim(0, h)
    axs[1, 1].invert_yaxis()
    axs[1, 1].set_aspect("equal", adjustable="box")
    # Only compute displacements if enough points are available
    if len(ref_pts) >= 2 and len(ab_pts) >= 2:
        # Estimate typical spot spacing to define a reasonable matching distance
        dists = cdist(ref_pts, ref_pts)
        np.fill_diagonal(dists, np.inf)
        median_spacing = np.median(np.min(dists, axis=1))
        max_dist = MAX_ASSIGN_FACTOR * median_spacing
        # Compute pairwise distances and find best one-to-one matches
        cost = cdist(ref_pts, ab_pts)
        r, c = linear_sum_assignment(cost)
        valid = cost[r, c] <= max_dist
        if np.any(valid):
            # Compute displacement vectors for valid matches
            dx = ab_pts[c[valid], 0] - ref_pts[r[valid], 0]
            dy = ab_pts[c[valid], 1] - ref_pts[r[valid], 1]
            # Draw displacement arrows
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
    # Save the figure if a path is provided
    if save_path:
        fig.savefig(save_path, dpi=200, facecolor="white")
    plt.close(fig)

# This function saves a diagnostic image showing spot matching and displacement
def save_diagnostic_png(res, output_dir):
    # Convert output directory to a Path object
    output_dir = Path(output_dir)
    # Use the frame name for naming the output file
    stem = res["name"]
    # Create a 2x2 figure for diagnostics
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.patch.set_facecolor("white")
    fig.suptitle(f"Displacement: {stem}", fontsize=12)
    # Show the reference input-plane image
    axs[0, 0].imshow(res["ref_img"], cmap="gray", origin="upper")
    axs[0, 0].set_title("Reference (IP)")
    axs[0, 0].axis("off")
    # Show the aberrated image
    axs[0, 1].imshow(res["ab_img"], cmap="gray", origin="upper")
    axs[0, 1].set_title("Aberrated (IA)")
    axs[0, 1].axis("off")
    # Overlay detected spot positions for reference and aberrated images
    axs[1, 0].imshow(np.zeros_like(res["ref_img"]), cmap="gray", origin="upper")
    axs[1, 0].scatter(res["ref_centroids"][:, 0], res["ref_centroids"][:, 1], s=8, c="lime")
    axs[1, 0].scatter(res["ab_centroids"][:, 0], res["ab_centroids"][:, 1], s=4, c="red")
    axs[1, 0].set_title("IP (green) vs IA (red)")
    axs[1, 0].set_aspect("equal", adjustable="box")
    axs[1, 0].axis("off")
    # Prepare the displacement vector plot
    axs[1, 1].set_title("Displacement Field")
    axs[1, 1].axis("off")
    h, w = res["ref_img"].shape
    axs[1, 1].set_xlim(0, w)
    axs[1, 1].set_ylim(0, h)
    axs[1, 1].invert_yaxis()
    axs[1, 1].set_aspect("equal", adjustable="box")
    # Draw arrows showing how each spot moved
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
    # Save the diagnostic image with settings optimized for speed
    fig.savefig(
        output_dir / f"displacement_{stem}.png",
        dpi=90,
        format="png",
        bbox_inches=None,
        facecolor="white",
        pil_kwargs={"compress_level": 3},
    )
    plt.close(fig)

# This function runs the full inference pipeline for one image pair (IP and IA)
def infer_pair(ref_image_path, ab_image_path, zernike_modes):
    global _A_cached, _A_pinv, _A_rows
    # Load reference and aberrated images in grayscale
    ref_img = cv2.imread(str(ref_image_path), cv2.IMREAD_GRAYSCALE)
    ab_img  = cv2.imread(str(ab_image_path), cv2.IMREAD_GRAYSCALE)
    # Stop if either image cannot be read
    if ref_img is None or ab_img is None:
        print(f"❌ Could not read images: {ref_image_path}")
        return None
    # Resize images and normalize for U-Net input
    ref_resized = cv2.resize(ref_img, (IMG_SIZE, IMG_SIZE)) / 255.0
    ab_resized  = cv2.resize(ab_img,  (IMG_SIZE, IMG_SIZE)) / 255.0
    # Run U-Net to predict spot masks
    ref_pred = model.predict(ref_resized[np.newaxis, ..., np.newaxis], verbose=0)[0, ..., 0]
    ab_pred  = model.predict(ab_resized[np.newaxis, ..., np.newaxis], verbose=0)[0, ..., 0]
    # Threshold and clean the predicted masks
    ref_mask = cleanup_mask((ref_pred > 0.45).astype(np.uint8) * 255)
    ab_mask  = cleanup_mask((ab_pred  > 0.45).astype(np.uint8) * 255)
    # Resize masks back to original image size
    ref_mask = cv2.resize(ref_mask, ref_img.shape[::-1], interpolation=cv2.INTER_NEAREST)
    ab_mask  = cv2.resize(ab_mask,  ab_img.shape[::-1],  interpolation=cv2.INTER_NEAREST)
    # Extract spot centroids from masks
    ref_pts = mask_to_centroids(ref_mask)
    ab_pts  = mask_to_centroids(ab_mask)
    # Fallback to classical masking if U-Net detects too few spots
    if len(ref_pts) < 50:
        ref_pts = mask_to_centroids(create_binary_mask_from_ip(ref_img))
    if len(ab_pts) < 50:
        ab_pts = mask_to_centroids(create_binary_mask_from_ip(ab_img))
    matched_ref = np.empty((0, 2))
    matched_ab  = np.empty((0, 2))
    displacements_px = np.empty((0, 2))
    # Match reference and aberrated spots using global assignment
    if len(ref_pts) >= 2 and len(ab_pts) >= 2:
        # Estimate typical spot spacing to define a safe matching distance
        dists = cdist(ref_pts, ref_pts)
        np.fill_diagonal(dists, np.inf)
        max_dist = MAX_ASSIGN_FACTOR * np.median(np.min(dists, axis=1))
        # Compute all pairwise distances and solve one-to-one matching
        cost = cdist(ref_pts, ab_pts)
        r, c = linear_sum_assignment(cost)
        valid = cost[r, c] <= max_dist
        # Keep only physically reasonable matches
        matched_ref = ref_pts[r[valid]]
        matched_ab  = ab_pts[c[valid]]
        # Compute pixel displacements for matched spots
        displacements_px = matched_ab - matched_ref
    zernike_coeffs = None
    wavefront = None
    wavefront_vis = None
    X = Y = None
    # Proceed only if enough slope measurements are available
    if len(displacements_px) >= len(zernike_modes):
        # Normalize centroid coordinates to the unit pupil
        x_norm, y_norm, R_px = normalize_centroids(matched_ref)
        # Convert pupil radius from pixels to physical units
        R_phys = R_px * PIXEL_PITCH
        # Convert centroid shifts into wavefront slopes
        b = -build_slope_vector(displacements_px * PIXEL_PITCH) / FOCAL_LENGTH
        # Reuse cached design matrix if possible for speed
        if _A_cached is not None and b.shape[0] == _A_rows:
            zernike_coeffs = _A_pinv @ b
        else:
            # Build the Zernike design matrix and solve least squares
            A = build_design_matrix(x_norm, y_norm, zernike_modes, R_phys)
            zernike_coeffs = solve_modal_coefficients(A, b)
            # Cache matrix and pseudo-inverse for future frames
            if _A_cached is None:
                _A_cached = A
                _A_pinv = np.linalg.pinv(A)
                _A_rows = A.shape[0]
                print("📌 Design matrix built and cached (safe)")
        # Reconstruct the wavefront on a dense grid
        X, Y, W_phys = reconstruct_wavefront(zernike_coeffs, zernike_modes)
        # Ensure arrays are on CPU for visualization
        from backend import to_cpu
        X, Y, W_phys, zernike_coeffs = map(to_cpu, (X, Y, W_phys, zernike_coeffs))
        # Prepare wavefront for visualization if valid values exist
        if np.isfinite(W_phys).any():
            wavefront = W_phys
            W_vis = (W_phys - np.nanmean(W_phys)) / 80e-9
            W_vis[(X**2 + Y**2) > 1.0] = 0.0
            wavefront_vis = np.flipud(W_vis).T
    # Return all results needed by visualization and diagnostics
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

# This function runs inference on all matching image pairs inside two folders
def infer_folder(ref_folder, ab_folder, output_dir):
    # Collect all valid image files from the reference folder
    ref_files = sorted(
        p for p in Path(ref_folder).iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    )
    # Stop if no images are found
    if not ref_files:
        print(f"❌ No image files found in {ref_folder}")
        return
    # Loop through each reference image
    for ref_path in ref_files:
        # Extract filename without extension
        stem = ref_path.stem
        # Skip files that do not follow the expected naming pattern
        if "_" not in stem:
            print(f"⚠️ Skipping unrecognized filename: {ref_path.name}")
            continue
        # Split name into prefix and frame index
        prefix, index = stem.rsplit("_", 1)
        # Ensure the index part is numeric
        if not index.isdigit():
            print(f"⚠️ Invalid index in filename: {ref_path.name}")
            continue
        # Construct the matching aberrated filename by replacing IP with IA
        ab_prefix = prefix.replace("IP", "IA", 1)
        ab_name = f"{ab_prefix}_{index}{ref_path.suffix}"
        ab_path = Path(ab_folder) / ab_name
        # Skip if the matching aberrated image does not exist
        if not ab_path.exists():
            print(f"⚠️ Missing aberrated file: {ab_name}")
            continue
        # Run inference on the matched image pair
        infer_pair(ref_path, ab_path, output_dir)

# This function finds matching reference and aberrated image pairs
def discover_pairs(ref_input, ab_input):
    frame_pairs = []
    # Case 1: both inputs are folders
    if os.path.isdir(ref_input) and os.path.isdir(ab_input):
        # Collect all valid image files from the reference folder
        ref_files = sorted(
            p for p in Path(ref_input).iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        )
        for ref_path in ref_files:
            stem = ref_path.stem
            # Skip files that do not follow the expected naming pattern
            if "_" not in stem:
                continue
            # Split name into prefix and frame index
            prefix, index = stem.rsplit("_", 1)
            if not index.isdigit():
                continue
            # Build the matching aberrated filename by replacing IP with IA
            ab_prefix = prefix.replace("IP", "IA", 1)
            ab_name = f"{ab_prefix}_{index}{ref_path.suffix}"
            ab_path = Path(ab_input) / ab_name
            # Keep the pair only if the aberrated file exists
            if ab_path.exists():
                frame_pairs.append((ref_path, ab_path))
    # Case 2: both inputs are single image files
    elif os.path.isfile(ref_input) and os.path.isfile(ab_input):
        frame_pairs.append((Path(ref_input), Path(ab_input)))
    # Invalid input combination
    else:
        raise ValueError("ref_input and ab_input must both be files or folders")
    return frame_pairs

# This function manages inference over one or more image pairs
def infer_manager(
    ref_input,
    ab_input,
    save_outputs=True,
    output_root=OUTPUT_DIR,
    n_zernike=10,
    emit_result=None,
):
    # Generate the list of Zernike modes to be used for reconstruction
    zernike_modes = generate_zernike_modes(n_zernike)
    # Create a new output folder if saving is enabled
    output_dir = make_timestamped_output_dir(output_root) if save_outputs else None
    # Discover all matching reference–aberrated image pairs
    frame_pairs = discover_pairs(ref_input, ab_input)
    print(f"▶ Processing {len(frame_pairs)} frame pairs (serial)")
    for idx, (ref_path, ab_path) in enumerate(frame_pairs):
        # Start timing this frame
        t0 = time.perf_counter()
        # Run inference on the current image pair
        res = infer_pair(ref_path, ab_path, zernike_modes)
        if res is None:
            continue
        # Send result immediately to live visualization if enabled
        if emit_result is not None:
            emit_result(res)
        # Save diagnostics asynchronously if requested
        if save_outputs:
            try:
                diagnostic_queue.put_nowait((res, output_dir))
            except Exception:
                pass
        # Print how long this frame took to process
        print(f"[Frame {idx:04d}] Compute: {(time.perf_counter() - t0)*1000:.2f} ms")

# Allow this file to be run directly from the command line
if __name__ == "__main__":
    import argparse
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description="U-Net inference for dot displacement")
    parser.add_argument("--ref", required=True)
    parser.add_argument("--ab", required=True)
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output directory (timestamped folder used if omitted)"
    )
    args = parser.parse_args()
    # Decide where outputs will be saved
    if args.out is None:
        output_dir = make_timestamped_output_dir(OUTPUT_DIR)
        print(f"📁 Auto output directory: {output_dir}")
    else:
        output_dir = args.out
        os.makedirs(output_dir, exist_ok=True)
    print("⏱️ Starting inference...")
    start_time = time.time()
    # If both inputs are folders, process all matching image pairs
    if os.path.isdir(args.ref) and os.path.isdir(args.ab):
        infer_folder(
            ref_folder=args.ref,
            ab_folder=args.ab,
            output_dir=output_dir
        )
    # If both inputs are single images, process just one pair
    elif os.path.isfile(args.ref) and os.path.isfile(args.ab):
        infer_pair(
            ref_image_path=args.ref,
            ab_image_path=args.ab,
            output_dir=output_dir
        )
    else:
        raise ValueError("ref and ab must both be folders or both be files")
    # Print total runtime
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"✅ Inference completed in {elapsed:.2f} seconds")
