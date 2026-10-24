"""Compare v1 and v2 on REAL field photos (PlantDoc val+test, held out from v1; test also from v2).

Accuracy is shown for a single pass and with test-time augmentation (4 views, as in the web app).
Run:  python compare_field.py        (needs model_out/krishi_daktar.keras and model_out_v2/krishi_daktar_v2.keras)
"""
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from config import *

MODELS = {"v1": ROOT / "model_out" / "krishi_daktar.keras",
          "v2": ROOT / "model_out_v2" / "krishi_daktar_v2.keras"}
PD = ROOT / "plantdoc_prepared"


def load(split):
    ds = keras.utils.image_dataset_from_directory(PD / split, label_mode="int", class_names=CLASSES,
                                                  image_size=(IMG_SIZE, IMG_SIZE), batch_size=64, shuffle=False)
    x = np.concatenate([a.numpy() for a, _ in ds]); y = np.concatenate([b.numpy() for _, b in ds])
    return x, y


def tta(model, x):
    """Average of original, mirrored and two centre-zoom views (same as the web app)."""
    def zoom(a, f):
        n = int(IMG_SIZE * f); o = (IMG_SIZE - n) // 2
        return tf.image.resize(a[:, o:o + n, o:o + n, :], (IMG_SIZE, IMG_SIZE)).numpy()
    views = [x, x[:, :, ::-1, :], zoom(x, 0.85), zoom(x, 0.7)]
    return np.mean([model.predict(v, verbose=0) for v in views], axis=0)


out = {}
for split in ("val", "test"):
    x, y = load(split)
    keep = np.isin(y, np.unique(y[np.array([len(list((PD / split / CLASSES[i]).glob("*.jpg"))) > 0 for i in y])]))
    x, y = x[keep], y[keep]
    for name, path in MODELS.items():
        if not path.exists():
            continue
        m = keras.models.load_model(path)
        p1 = m.predict(x, verbose=0); p2 = tta(m, x)
        out[f"{name}_{split}"] = {"n": int(len(y)),
                                  "single": float((p1.argmax(1) == y).mean()),
                                  "tta": float((p2.argmax(1) == y).mean())}
        print(name, split, out[f"{name}_{split}"])
json.dump(out, open(ROOT / "results_field.json", "w"), indent=2)
