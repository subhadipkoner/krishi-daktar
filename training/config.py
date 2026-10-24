"""Shared settings for Krishi Daktar training. Edit paths/hyperparameters here only."""
import os
from pathlib import Path

VERSION = os.environ.get("KD_VERSION", "")   # "" = v1, "_v2" = field-augmented model

ROOT = Path(__file__).resolve().parent.parent          # project root (or /content/bundle on Colab)
PV = ROOT / "PlantVillage"                              # PlantVillage top-level class folders
RICE = ROOT / "rice_leaf_diseases"
NEW = ROOT / "new_data"                                 # sugarcane (Kaggle: Sugarcane Leaf Disease Dataset, CC0)                      # rice dataset (one folder per class)

# class_id -> source folder. Pepper is skipped (out of scope).
# The nested PlantVillage/PlantVillage duplicate folder is ignored.
SOURCES = {
    "potato_early_blight": PV / "Potato___Early_blight",
    "potato_late_blight": PV / "Potato___Late_blight",
    "potato_healthy": PV / "Potato___healthy",
    "tomato_bacterial_spot": PV / "Tomato_Bacterial_spot",
    "tomato_early_blight": PV / "Tomato_Early_blight",
    "tomato_late_blight": PV / "Tomato_Late_blight",
    "tomato_leaf_mold": PV / "Tomato_Leaf_Mold",
    "tomato_septoria_leaf_spot": PV / "Tomato_Septoria_leaf_spot",
    "tomato_spider_mites": PV / "Tomato_Spider_mites_Two_spotted_spider_mite",
    "tomato_target_spot": PV / "Tomato__Target_Spot",
    "tomato_mosaic_virus": PV / "Tomato__Tomato_mosaic_virus",
    "tomato_yellow_leaf_curl_virus": PV / "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "tomato_healthy": PV / "Tomato_healthy",
    "rice_bacterial_leaf_blight": RICE / "Bacterial leaf blight",
    "rice_brown_spot": RICE / "Brown spot",
    "rice_leaf_smut": RICE / "Leaf smut",
    "sugarcane_healthy": NEW / "Healthy",
    "sugarcane_mosaic": NEW / "Mosaic",
    "sugarcane_red_rot": NEW / "RedRot",
    "sugarcane_rust": NEW / "Rust",
    "sugarcane_yellow_leaf": NEW / "Yellow",
}
# Alphabetical order == the order Keras uses for folder labels. Do not change.
CLASSES = sorted(SOURCES)

IMG_SIZE = 224            # MobileNetV2 input
PREP_SIZE = 256           # size images are resized to when preparing (smaller zip, faster training)
BATCH = 32
SEED = 42
MAX_PER_CLASS = 1200      # cap big classes so training is balanced and fast
SPLIT = (0.70, 0.15, 0.15)  # train / val / test

PREPARED = ROOT / "dataset_prepared"
RESULTS = ROOT / f"results{VERSION}"
MODEL_DIR = ROOT / f"model_out{VERSION}"
WEBAPP_MODEL = ROOT / "webapp" / f"model{VERSION}"
LABELS_JSON = ROOT / "webapp" / "data" / "labels.json"
