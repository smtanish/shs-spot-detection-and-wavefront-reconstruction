
import os

# Paths assume this script is in the same folder as 'nunoisyimagesset'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "nunoisyimagesset")
TRAIN_FOLDER = os.path.join(DATA_ROOT, "IP_NU_noisy", "IP_training")

MODEL_PATH = os.path.join(BASE_DIR, "unet_trained.keras")
missing = [p for p in (TRAIN_FOLDER,) if not os.path.exists(p)]
if missing:
    raise FileNotFoundError(f"Missing required training folder(s): {missing}")

import os
import cv2
import numpy as np
import tensorflow as tf
from keras import layers, models



# Configuration
IMG_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 6
MAX_ASSIGN_FACTOR = 0.6
SEED = 42

# Reproducibility
tf.random.set_seed(SEED)
np.random.seed(SEED)

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
    return out
def build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 1)):
    """Defines a small U-Net suitable for Colab resources."""
    inputs = layers.Input(shape=input_shape)

    # Encoder path
    c1 = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    c1 = layers.Conv2D(32, 3, padding='same', activation='relu')(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(64, 3, padding='same', activation='relu')(p1)
    c2 = layers.Conv2D(64, 3, padding='same', activation='relu')(c2)
    p2 = layers.MaxPooling2D()(c2)

    c3 = layers.Conv2D(128, 3, padding='same', activation='relu')(p2)
    c3 = layers.Conv2D(128, 3, padding='same', activation='relu')(c3)
    p3 = layers.MaxPooling2D()(c3)

    # Bottleneck
    b = layers.Conv2D(256, 3, padding='same', activation='relu')(p3)
    b = layers.Conv2D(256, 3, padding='same', activation='relu')(b)

    # Decoder path
    u3 = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(b)
    u3 = layers.Concatenate()([u3, c3])
    u3 = layers.Conv2D(128, 3, padding='same', activation='relu')(u3)
    u3 = layers.Conv2D(128, 3, padding='same', activation='relu')(u3)

    u2 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(u3)
    u2 = layers.Concatenate()([u2, c2])
    u2 = layers.Conv2D(64, 3, padding='same', activation='relu')(u2)
    u2 = layers.Conv2D(64, 3, padding='same', activation='relu')(u2)

    u1 = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(u2)
    u1 = layers.Concatenate()([u1, c1])
    u1 = layers.Conv2D(32, 3, padding='same', activation='relu')(u1)
    u1 = layers.Conv2D(32, 3, padding='same', activation='relu')(u1)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(u1)

    model = models.Model(inputs, outputs)
    return model
def load_ip_images_from(folder):
    images_resized = []
    masks_resized = []
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".png"):
            continue
        full_path = os.path.join(folder, fname)
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        mask = create_binary_mask_from_ip(img)
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        mask_resized = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        images_resized.append((img_resized.astype(np.float32) / 255.0)[..., np.newaxis])
        masks_resized.append((mask_resized.astype(np.float32) / 255.0)[..., np.newaxis])
    return images_resized, masks_resized


# Define training and test directories (already defined earlier)
# Just reuse the TRAIN_FOLDER and TEST_FOLDER variables

import time

# Load ONLY training data
ip_train_images, ip_train_masks = load_ip_images_from(TRAIN_FOLDER)

# Convert to NumPy
X_train = np.stack(ip_train_images, axis=0)
y_train = np.stack(ip_train_masks, axis=0)

# Create training dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(
    buffer_size=len(X_train)
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Augmentation
def augment(x, y):
    x = tf.image.random_flip_left_right(x)
    y = tf.image.random_flip_left_right(y)
    x = tf.image.random_flip_up_down(x)
    y = tf.image.random_flip_up_down(y)
    return x, y

train_dataset = train_dataset.map(augment)

# Timing callback
class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self.epoch_time_start
        self.times.append(elapsed)
        print(f"🕒 Epoch {epoch+1} took {elapsed:.2f} seconds.")


from keras.models import load_model

if os.path.exists(MODEL_PATH):
    print(f"📦 Loading existing trained model from {MODEL_PATH}")
    model = load_model(MODEL_PATH)

else:
    print("🚀 No trained model found. Training U-Net...")

    model = build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 1))
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    time_callback = TimeHistory()
    start_time = time.time()

    model.fit(
        train_dataset,
        epochs=EPOCHS,
        callbacks=[time_callback]
    )

    model.save(MODEL_PATH)
    print(f"💾 Model saved to {MODEL_PATH}")