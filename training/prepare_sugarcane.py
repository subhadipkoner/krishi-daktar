"""Add the sugarcane classes: split 70/15/15, resize to 224, write labels.json (all classes) and
small zip parts (< 9 MB each) that can be uploaded to Colab through the Files panel."""
import json, random, shutil, zipfile
from PIL import Image, ImageOps
from config import *

NEW_IDS = [c for c in CLASSES if c.startswith("sugarcane_")]
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
OUT = ROOT / "sugarcane_prepared"


def main():
    random.seed(SEED)
    if OUT.exists():
        shutil.rmtree(OUT)
    for cid in NEW_IDS:
        files = sorted(p for p in SOURCES[cid].rglob("*") if p.suffix.lower() in EXT)
        random.shuffle(files)
        files = files[:MAX_PER_CLASS]
        n = len(files); a, b = int(n * SPLIT[0]), int(n * (SPLIT[0] + SPLIT[1]))
        parts = {"train": files[:a], "val": files[a:b], "test": files[b:]}
        for split, items in parts.items():
            d = OUT / "dataset_prepared" / split / cid
            d.mkdir(parents=True, exist_ok=True)
            for i, f in enumerate(items):
                ImageOps.exif_transpose(Image.open(f)).convert("RGB").resize((IMG_SIZE, IMG_SIZE)).save(d / f"{i:05d}.jpg", quality=88)
        print(cid, {k: len(v) for k, v in parts.items()})
    # label map for the web app: index follows sorted CLASSES (same order Keras uses)
    LABELS_JSON.write_text(json.dumps([{"index": i, "id": c, "crop": c.split("_")[0]} for i, c in enumerate(CLASSES)], indent=2))
    z = ROOT / "sugarcane_prepared.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zf:
        for p in sorted(OUT.rglob("*.jpg")):
            zf.write(p, p.relative_to(OUT).as_posix())
    data = z.read_bytes() if hasattr(z, "read_bytes") else open(z, "rb").read()
    chunk = 8 * 1024 * 1024
    for i in range(0, len(data), chunk):
        (ROOT / f"sugar_part_{i // chunk:02d}").write_bytes(data[i:i + chunk])
    print("zip MB", round(len(data) / 1e6, 1), "parts", -(-len(data) // chunk), "| classes total", len(CLASSES))


if __name__ == "__main__":
    main()
