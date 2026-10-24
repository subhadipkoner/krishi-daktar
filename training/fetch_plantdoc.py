"""Fetch REAL field photos (PlantDoc, CC BY 4.0, github.com/pratikkayal/PlantDoc-Dataset).

Run inside Colab (the download happens cloud-to-cloud). Only our tomato and potato folders are
fetched (git sparse checkout, about 300 MB). Output:
  plantdoc_prepared/{train,val,test}/<class_id>/*.jpg   (256 px, same ids as the main dataset)
train = 70% of PlantDoc train (used in v2 training), val = 30% of PlantDoc train (model selection),
test = PlantDoc's own test split (held-out REAL field test, never trained on).
"""
import random, shutil, subprocess
from pathlib import Path
from PIL import Image, ImageOps
from config import ROOT, CLASSES, PREP_SIZE, SEED

MAP = {  # PlantDoc folder -> our class id
    "Potato leaf early blight": "potato_early_blight",
    "Potato leaf late blight": "potato_late_blight",
    "Tomato Early blight leaf": "tomato_early_blight",
    "Tomato Septoria leaf spot": "tomato_septoria_leaf_spot",
    "Tomato leaf": "tomato_healthy",
    "Tomato leaf bacterial spot": "tomato_bacterial_spot",
    "Tomato leaf late blight": "tomato_late_blight",
    "Tomato leaf mosaic virus": "tomato_mosaic_virus",
    "Tomato leaf yellow virus": "tomato_yellow_leaf_curl_virus",
    "Tomato mold leaf": "tomato_leaf_mold",
}
RAW = ROOT / "plantdoc_raw"
OUT = ROOT / "plantdoc_prepared"


def main():
    if RAW.exists():
        shutil.rmtree(RAW)
    run = lambda *a, **k: subprocess.run(*a, check=True, **k)
    run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
         "https://github.com/pratikkayal/PlantDoc-Dataset", str(RAW)])
    run(["git", "-C", str(RAW), "sparse-checkout", "set", "--no-cone"] +
        [f"/{s}/{name}/" for s in ("train", "test") for name in MAP])
    if OUT.exists():
        shutil.rmtree(OUT)
    for s in ("train", "val", "test"):
        for c in CLASSES:
            (OUT / s / c).mkdir(parents=True, exist_ok=True)   # empty folders are fine
    random.seed(SEED)
    counts = {}
    for name, cid in MAP.items():
        tr = sorted((RAW / "train" / name).glob("*"))
        random.shuffle(tr)
        k = int(len(tr) * 0.7)
        parts = {"train": tr[:k], "val": tr[k:], "test": sorted((RAW / "test" / name).glob("*"))}
        for s, files in parts.items():
            n = 0
            for i, f in enumerate(files):
                try:
                    img = ImageOps.exif_transpose(Image.open(f)).convert("RGB").resize((PREP_SIZE, PREP_SIZE))
                    img.save(OUT / s / cid / f"{i:04d}.jpg", quality=90); n += 1
                except Exception as e:
                    print("skip", f.name, e)
            counts[(cid, s)] = n
    for cid in sorted(set(MAP.values())):
        print(f"{cid:32s}", {s: counts.get((cid, s), 0) for s in ("train", "val", "test")})
    shutil.rmtree(RAW)


if __name__ == "__main__":
    main()
