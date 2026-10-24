"""Step 3: evaluate on the held-out TEST split. Saves confusion matrix + per-class accuracy plots."""
import json
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report
from config import *


def main():
    model = keras.models.load_model(MODEL_DIR / f"krishi_daktar{VERSION}.keras")
    test = keras.utils.image_dataset_from_directory(
        PREPARED / "test", label_mode="int", class_names=CLASSES,
        image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH, shuffle=False)
    y_true = np.concatenate([y.numpy() for _, y in test])
    probs = model.predict(test)
    y_pred = probs.argmax(1)

    cm = confusion_matrix(y_true, y_pred)
    per_class = cm.diagonal() / cm.sum(1)
    overall = float((y_true == y_pred).mean())
    top2 = float(np.mean([t in np.argsort(p)[-2:] for t, p in zip(y_true, probs)]))
    # how often is the model confident (>=70%)? and how accurate is it when it is?
    conf = probs.max(1); sure = conf >= 0.70
    acc_sure = float((y_true[sure] == y_pred[sure]).mean()) if sure.any() else 0.0

    RESULTS.mkdir(exist_ok=True)
    # confusion matrix
    plt.figure(figsize=(11, 9))
    plt.imshow(cm, cmap="Greens"); plt.colorbar()
    plt.xticks(range(len(CLASSES)), CLASSES, rotation=90, fontsize=7)
    plt.yticks(range(len(CLASSES)), CLASSES, fontsize=7)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if cm[i, j]:
                plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(f"Confusion matrix (test accuracy {overall:.1%})")
    plt.tight_layout(); plt.savefig(RESULTS / "confusion_matrix.png", dpi=150); plt.close()

    # per-class accuracy
    order = np.argsort(per_class)
    plt.figure(figsize=(8, 6))
    plt.barh([CLASSES[i] for i in order], per_class[order] * 100, color="seagreen")
    plt.xlabel("Accuracy (%)"); plt.xlim(0, 100); plt.title("Per-class test accuracy")
    plt.tight_layout(); plt.savefig(RESULTS / "per_class_accuracy.png", dpi=150); plt.close()

    metrics = {"overall_accuracy": overall, "top2_accuracy": top2,
               "accuracy_when_confidence_ge_70": acc_sure, "fraction_confident": float(sure.mean()),
               "test_images": int(len(y_true)),
               "per_class_accuracy": {c: float(a) for c, a in zip(CLASSES, per_class)},
               "per_class_test_count": {c: int(n) for c, n in zip(CLASSES, cm.sum(1))}}
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=3))
    print(json.dumps({k: v for k, v in metrics.items() if not isinstance(v, dict)}, indent=2))


if __name__ == "__main__":
    main()
