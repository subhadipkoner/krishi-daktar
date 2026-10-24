"""Step 1: build a clean train/val/test split (resized to 256px) from the raw datasets.

Run once locally:  python prepare_data.py
Output: dataset_prepared/{train,val,test}/<class_id>/*.jpg, webapp/data/labels.json,
        and colab_bundle.zip (dataset + training scripts) to upload to Google Drive.
"""
import json, random, shutil, zipfile
from PIL import Image, ImageOps
from config import *

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    random.seed(SEED)
    if PREPARED.exists():
        shutil.rmtree(PREPARED)
    report = {}
    for cid in CLASSES:
        src = SOURCES[cid]
        files = sorted(p for p in src.rglob("*") if p.suffix.lower() in EXT)
        if not files:
            raise SystemExit(f"No images found for {cid} in {src}")
        random.shuffle(files)
        files = files[:MAX_PER_CLASS]
        n = len(files)
        n_tr, n_va = int(n * SPLIT[0]), int(n * SPLIT[1])
        parts = {"train": files[:n_tr], "val": files[n_tr:n_tr + n_va], "test": files[n_tr + n_va:]}
        for split, items in parts.items():
            out = PREPARED / split / cid
            out.mkdir(parents=True, exist_ok=True)
            for i, f in enumerate(items):
                img = ImageOps.exif_transpose(Image.open(f)).convert("RGB").resize((PREP_SIZE, PREP_SIZE))
                img.save(out / f"{i:05d}.jpg", quality=90)
        report[cid] = {k: len(v) for k, v in parts.items()}
        print(cid, report[cid])

    # label mapping used by the web app (index -> class id)
    LABELS_JSON.parent.mkdir(parents=True, exist_ok=True)
    LABELS_JSON.write_text(json.dumps(
        [{"index": i, "id": c, "crop": c.split("_")[0]} for i, c in enumerate(CLASSES)], indent=2))

    # zip dataset + scripts for Colab
    zpath = ROOT / "colab_bundle.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
        for p in PREPARED.rglob("*.jpg"):
            z.write(p, Path("bundle") / p.relative_to(ROOT))
        for p in (ROOT / "training").glob("*.py"):
            z.write(p, Path("bundle/training") / p.name)
        z.write(LABELS_JSON, "bundle/webapp/data/labels.json")
    print(f"\nWrote {zpath} ({zpath.stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
