# This file trains a U-Net cnn to detect bright Shack–Hartmann spots.
# It takes noisy input-plane images, generates binary spot masks using simple image processing, and uses those masks to train the network.
# The trained model learns what valid spots look like even under noise and optical distortion, and is later used during inference to segment spots before centroid extraction.

import os
# Get the folder where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Main dataset folder containing all training data
DATA_ROOT = os.path.join(BASE_DIR, "nunoisyimagesset")
# Folder that holds the noisy input-plane images used for training
TRAIN_FOLDER = os.path.join(DATA_ROOT, "IP_NU_noisy", "IP_training")
# Location where the trained U-Net model will be saved
MODEL_PATH = os.path.join(BASE_DIR, "unet_trained.keras")
# Stop execution early if the training data folder is missing
missing = [p for p in (TRAIN_FOLDER,) if not os.path.exists(p)]
if missing:
    raise FileNotFoundError(f"Missing required training folder(s): {missing}")
# Import libraries for image processing and neural network training
import os
import cv2
import numpy as np
import tensorflow as tf
from keras import layers, models
# Size to which all images are resized before training
IMG_SIZE = 128
# Number of images processed together during training
BATCH_SIZE = 8
# Number of times the model sees the full dataset
EPOCHS = 6
# Factor used later to decide how far a spot is allowed to move before it is considered a bad match
# It is defined here to keep training and inference settings consistent
MAX_ASSIGN_FACTOR = 0.6
# Fixed random seed so training behaves the same each time this script is run
SEED = 42

# Set fixed seeds so results stay the same across different runs
tf.random.set_seed(SEED)
np.random.seed(SEED)

# This function compensates for brightness falloff from the center to the edges of the image
def radial_compensate(img):
    # Get image height and width
    h, w = img.shape
    # Compute image center
    cy, cx = h // 2, w // 2
    # Create coordinate grids for distance calculation
    Y, X = np.ogrid[:h, :w]
    # Compute distance of each pixel from the image center
    r = np.sqrt((X - cx)**2 + (Y - cy)**2)
    # Normalize radius so it ranges from 0 (center) to 1 (edge)
    r_norm = r / r.max()
    # Number of radial rings used to estimate brightness profile
    nbins = 80
    bins = np.linspace(0.0, 1.0, nbins + 1)
    # Store median brightness per radial ring
    median_profile = np.zeros(nbins)
    for i in range(nbins):
        # Select pixels that fall inside the current radial ring
        mask = (r_norm >= bins[i]) & (r_norm < bins[i+1])
        # Use median brightness to reduce noise sensitivity
        median_profile[i] = np.median(img[mask]) if np.any(mask) else median_profile[i-1] if i > 0 else 1.0
    # Smooth the radial brightness profile to avoid sharp jumps
    median_profile = cv2.blur(median_profile.reshape(-1,1), (5,1)).ravel()
    # Prevent division by very small values
    median_profile = np.maximum(median_profile, 1e-4)
    # Compute gain needed to equalize brightness across radius
    gain_profile = median_profile.max() / median_profile
    # Convert the 1D radial gain into a full image-sized gain map
    gain_image = np.interp(r_norm.ravel(), 0.5*(bins[:-1]+bins[1:]), gain_profile).reshape(h,w)
    # Limit gain to avoid over-amplifying noise near the edges
    gain_image = np.clip(gain_image, 1.0, 4.5)
    # Apply gain and clamp output to valid intensity range
    compensated = np.clip(img * gain_image, 0.0, 1.0)
    return compensated, gain_image

# This function creates a clean binary mask of bright spots from an input-plane image
def create_binary_mask_from_ip(ip_img, debug_show=False):
    # Convert image to float and normalize pixel values to 0–1
    imgf = ip_img.astype(np.float32) / 255.0
    # Compensate for radial brightness falloff
    comp, gain = radial_compensate(imgf)
    # Blur slightly to reduce noise and smooth spot shapes
    blur = cv2.GaussianBlur((comp*255).astype(np.uint8), (5,5), 0)
    # Automatically threshold the image to separate spots from background
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Create a small circular kernel for morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    # Fill small holes inside detected spots
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    # Remove small isolated noise blobs
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    # Label all connected bright regions
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(th, connectivity=8)
    # Create an empty output mask
    out = np.zeros_like(th)
    # Minimum pixel area required to keep a detected spot
    min_area = 2
    for lab in range(1, num_labels):
        # Keep only regions large enough to be real spots
        if stats[lab, cv2.CC_STAT_AREA] >= min_area:
            out[labels == lab] = 255
    return out

# This function builds the U-Net model used to segment bright spots in the images
def build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 1)):
    # Define the input layer with the expected image shape
    inputs = layers.Input(shape=input_shape)
    # First encoder block: learn basic edges and textures
    c1 = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    c1 = layers.Conv2D(32, 3, padding='same', activation='relu')(c1)
    # Reduce image size while keeping important features
    p1 = layers.MaxPooling2D()(c1)
    # Second encoder block: learn more complex patterns
    c2 = layers.Conv2D(64, 3, padding='same', activation='relu')(p1)
    c2 = layers.Conv2D(64, 3, padding='same', activation='relu')(c2)
    p2 = layers.MaxPooling2D()(c2)
    # Third encoder block: capture higher-level spot structure
    c3 = layers.Conv2D(128, 3, padding='same', activation='relu')(p2)
    c3 = layers.Conv2D(128, 3, padding='same', activation='relu')(c3)
    p3 = layers.MaxPooling2D()(c3)
    # Bottleneck: most compressed representation of the image
    b = layers.Conv2D(256, 3, padding='same', activation='relu')(p3)
    b = layers.Conv2D(256, 3, padding='same', activation='relu')(b)
    # First decoder step: upsample and combine with encoder features
    u3 = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(b)
    u3 = layers.Concatenate()([u3, c3])
    u3 = layers.Conv2D(128, 3, padding='same', activation='relu')(u3)
    u3 = layers.Conv2D(128, 3, padding='same', activation='relu')(u3)
    # Second decoder step
    u2 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(u3)
    u2 = layers.Concatenate()([u2, c2])
    u2 = layers.Conv2D(64, 3, padding='same', activation='relu')(u2)
    u2 = layers.Conv2D(64, 3, padding='same', activation='relu')(u2)
    # Final decoder step to restore original image resolution
    u1 = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(u2)
    u1 = layers.Concatenate()([u1, c1])
    u1 = layers.Conv2D(32, 3, padding='same', activation='relu')(u1)
    u1 = layers.Conv2D(32, 3, padding='same', activation='relu')(u1)
    # Output layer producing a probability mask for spot locations
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(u1)
    # Build and return the model
    model = models.Model(inputs, outputs)
    return model

# This function loads training images and creates corresponding binary masks
def load_ip_images_from(folder):
    images_resized = []
    masks_resized = []
    # Loop through all image files in the training folder
    for fname in sorted(os.listdir(folder)):
        # Only process PNG images
        if not fname.endswith(".png"):
            continue
        full_path = os.path.join(folder, fname)
        # Load image in grayscale
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        # Skip files that fail to load
        if img is None:
            continue
        # Generate a binary spot mask from the image
        mask = create_binary_mask_from_ip(img)
        # Resize image to match U-Net input size
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        # Resize mask using nearest neighbor to keep it binary
        mask_resized = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        # Normalize image to 0–1 and add channel dimension
        images_resized.append((img_resized.astype(np.float32) / 255.0)[..., np.newaxis])
        # Normalize mask to 0–1 and add channel dimension
        masks_resized.append((mask_resized.astype(np.float32) / 255.0)[..., np.newaxis])
    return images_resized, masks_resized

# Import time module for training timing
import time
# Load only the training images and masks
ip_train_images, ip_train_masks = load_ip_images_from(TRAIN_FOLDER)
# Convert lists into NumPy arrays for TensorFlow
X_train = np.stack(ip_train_images, axis=0)
y_train = np.stack(ip_train_masks, axis=0)
# Create a TensorFlow dataset from images and masks
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
# Shuffle the data, group it into batches, and prepare it for fast loading
train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# This function applies simple data augmentation to make training more robust
def augment(x, y):
    # Randomly flip images left–right
    x = tf.image.random_flip_left_right(x)
    y = tf.image.random_flip_left_right(y)
    # Randomly flip images up–down
    x = tf.image.random_flip_up_down(x)
    y = tf.image.random_flip_up_down(y)
    return x, y

# Apply augmentation to the training dataset
train_dataset = train_dataset.map(augment)

# This callback measures how long each training epoch takes
class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        # Store timing results for all epochs
        self.times = []
    def on_epoch_begin(self, epoch, logs=None):
        # Record start time of the current epoch
        self.epoch_time_start = time.time()
    def on_epoch_end(self, epoch, logs=None):
        # Compute and print how long the epoch took
        elapsed = time.time() - self.epoch_time_start
        self.times.append(elapsed)
        print(f"🕒 Epoch {epoch+1} took {elapsed:.2f} seconds.")

from keras.models import load_model
# Check if a trained model already exists on disk
if os.path.exists(MODEL_PATH):
    # Load the existing model to avoid retraining
    print(f"📦 Loading existing trained model from {MODEL_PATH}")
    model = load_model(MODEL_PATH)
else:
    # Train a new model if none is found
    print("🚀 No trained model found. Training U-Net...")
    # Build the U-Net architecture
    model = build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 1))
    # Configure how the model learns during training
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    # Print a summary of the model structure
    model.summary()
    # Create callback to track training time per epoch
    time_callback = TimeHistory()
    start_time = time.time()
    # Train the model on the prepared dataset
    model.fit(
        train_dataset,
        epochs=EPOCHS,
        callbacks=[time_callback]
    )
    # Save the trained model for later inference use
    model.save(MODEL_PATH)
    print(f"💾 Model saved to {MODEL_PATH}")
