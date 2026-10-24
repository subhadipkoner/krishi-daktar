"""Step 2: train MobileNetV2 (transfer learning) with heavy field-style augmentation.

Run:  python train.py            (needs tensorflow; GPU recommended -> use Colab)
Phase 1: train only the new classifier head (base frozen).
Phase 2: fine-tune the top layers of MobileNetV2 with a small learning rate.
"""
import json
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from config import *

tf.keras.utils.set_random_seed(SEED)


PD = ROOT / "plantdoc_prepared"                             # real field photos (see fetch_plantdoc.py)
PD_REPEAT = 3                                                # PlantDoc is small, so repeat it in training


def load_pd(split, shuffle):
    return keras.utils.image_dataset_from_directory(
        PD / split, labels="inferred", label_mode="int", class_names=CLASSES,
        image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH, shuffle=shuffle, seed=SEED)


def load(split, shuffle):
    return keras.utils.image_dataset_from_directory(
        PREPARED / split, labels="inferred", label_mode="int", class_names=CLASSES,
        image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH, shuffle=shuffle, seed=SEED)


# ---------- augmentation (only used on training data) ----------
# Fast, pure TensorFlow ops (no Keras layers) so it runs safely inside tf.data on any Keras version.
def geo_color(x):
    """Random flip, rotation, zoom, shift, brightness, contrast = different phone angles/lighting."""
    b = tf.shape(x)[0]
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    angle = tf.random.uniform([b], -0.6, 0.6)                    # radians (about +-34 degrees)
    zoom = tf.random.uniform([b], 0.75, 1.2)                     # >1 zooms in
    shift = tf.random.uniform([b, 2], -0.1, 0.1) * IMG_SIZE
    c, s_ = tf.cos(angle) / zoom, tf.sin(angle) / zoom
    cx = IMG_SIZE / 2.0
    transforms = tf.stack([c, -s_, cx - c * cx + s_ * cx + shift[:, 0],
                           s_, c, cx - s_ * cx - c * cx + shift[:, 1],
                           tf.zeros([b]), tf.zeros([b])], axis=1)
    x = tf.raw_ops.ImageProjectiveTransformV3(
        images=x, transforms=transforms, output_shape=[IMG_SIZE, IMG_SIZE],
        interpolation="BILINEAR", fill_mode="REFLECT", fill_value=0.0)
    x = x * tf.random.uniform([b, 1, 1, 1], 0.6, 1.4)            # brightness (sun / shade)
    mean = tf.reduce_mean(x, axis=[1, 2, 3], keepdims=True)
    x = (x - mean) * tf.random.uniform([b, 1, 1, 1], 0.7, 1.3) + mean   # contrast
    return x


def random_blur(x):
    """30% of images get a 5x5 box blur (shaky / out-of-focus phone photo)."""
    blurred = tf.nn.avg_pool2d(x, 5, 1, "SAME")
    use = tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) < 0.3
    return tf.where(use, blurred, x)


def random_background(x):
    """Lab images have plain grey/black/white backgrounds. Replace low-saturation pixels
    (the background) with random coloured texture + noise in 50% of images, so the model
    does not learn 'plain background = healthy'."""
    mx, mn = tf.reduce_max(x, -1, keepdims=True), tf.reduce_min(x, -1, keepdims=True)
    mask = tf.cast(((mx - mn) / (mx + 1e-6) < 0.15) | (mx < 40), tf.float32)
    b = tf.shape(x)[0]
    colour = tf.random.uniform([b, 1, 1, 3], 0, 255)
    texture = tf.image.resize(tf.random.uniform([b, 8, 8, 3], 0, 255), (IMG_SIZE, IMG_SIZE), "bicubic")
    bg = tf.clip_by_value(0.5 * colour + 0.5 * texture + tf.random.normal(tf.shape(x), 0, 25), 0, 255)
    use = tf.cast(tf.random.uniform([b, 1, 1, 1]) < 0.5, tf.float32)
    m = mask * use
    return x * (1 - m) + bg * m


def augment(x, y):
    x = random_background(x)
    x = geo_color(x)
    x = random_blur(x)
    return tf.clip_by_value(x, 0, 255), y


def build_model():
    base = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
    base.trainable = False
    inp = keras.Input((IMG_SIZE, IMG_SIZE, 3))              # raw pixels 0..255
    x = layers.Rescaling(1 / 127.5, offset=-1)(inp)         # MobileNetV2 wants -1..1 (inside the model,
    x = base(x, training=False)                             #  so the web app only sends 0..255 pixels)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(len(CLASSES), activation="softmax")(x)
    return keras.Model(inp, out), base


def main():
    RESULTS.mkdir(exist_ok=True); MODEL_DIR.mkdir(exist_ok=True)
    train, val = load("train", True), load("val", False)
    pd_counts = np.zeros(len(CLASSES))
    if False and PD.exists():                                           # mix real field photos into training / validation
        pd_counts = np.array([len(list((PD / "train" / c).glob("*.jpg"))) for c in CLASSES]) * PD_REPEAT
        pd_train = load_pd("train", True).unbatch().repeat(PD_REPEAT)
        train = train.unbatch().concatenate(pd_train).shuffle(4000, seed=SEED).batch(BATCH)
        val = val.unbatch().concatenate(load_pd("val", False).unbatch()).batch(BATCH)
        print("PlantDoc field photos mixed in:", int(pd_counts.sum() / PD_REPEAT), "train images x", PD_REPEAT)

    # class weights: rare classes (e.g. rice, healthy potato) count more
    counts = np.array([len(list((PREPARED / "train" / c).glob("*.jpg"))) for c in CLASSES]) + pd_counts
    cw = tf.constant([counts.sum() / (len(CLASSES) * n) for n in counts], dtype=tf.float32)

    def add_weight(x, y):                                   # rare class -> bigger weight
        x, y = augment(x, y)
        return x, y, tf.gather(cw, y)

    train = train.map(add_weight, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    val = val.prefetch(tf.data.AUTOTUNE)

    model, base = build_model()
    cbs = [keras.callbacks.EarlyStopping("val_accuracy", patience=4, restore_best_weights=True),
           keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.3, patience=2)]

    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h1 = model.fit(train, validation_data=val, epochs=5, callbacks=cbs)

    base.trainable = True
    for layer in base.layers[:-50]:                         # keep early layers frozen
        layer.trainable = False
    model.compile(optimizer=keras.optimizers.Adam(1e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h2 = model.fit(train, validation_data=val, epochs=8, callbacks=cbs)

    model.save(MODEL_DIR / f"krishi_daktar{VERSION}.keras")

    # training curves for the presentation
    acc = h1.history["accuracy"] + h2.history["accuracy"]
    vacc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    plt.figure(figsize=(7, 4))
    plt.plot(acc, label="train"); plt.plot(vacc, label="validation")
    plt.axvline(len(h1.history["accuracy"]) - 1, ls="--", c="gray", label="start fine-tuning")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend(); plt.title("Training curve")
    plt.tight_layout(); plt.savefig(RESULTS / "training_curve.png", dpi=150)
    print("Saved model to", MODEL_DIR)


if __name__ == "__main__":
    main()
