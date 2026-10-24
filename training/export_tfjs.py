"""Step 4: export the trained Keras model to TensorFlow.js (tfjs_graph_model) for the web app.

Needs:  pip install tensorflowjs
Output: webapp/model/model.json + weight shards (about 9 MB for MobileNetV2).
"""
import shutil, sys
import numpy as np
# tensorflowjs still uses old numpy aliases (np.object ...) that numpy 2 removed -> add them back
for _n, _t in (('object', object), ('bool', bool), ('int', int), ('float', float), ('str', str)):
    if not hasattr(np, _n):
        setattr(np, _n, _t)
from tensorflow import keras
from config import *


def main():
    model = keras.models.load_model(MODEL_DIR / f"krishi_daktar{VERSION}.keras")
    saved = MODEL_DIR / "saved_model"
    if saved.exists():
        shutil.rmtree(saved)
    model.export(str(saved))                                   # TF SavedModel
    if WEBAPP_MODEL.exists():
        shutil.rmtree(WEBAPP_MODEL)
    WEBAPP_MODEL.parent.mkdir(parents=True, exist_ok=True)
    from tensorflowjs.converters import converter       # same as the tensorflowjs_converter command
    sys.argv = ["tensorflowjs_converter", "--input_format=tf_saved_model", "--output_format=tfjs_graph_model",
                "--signature_name=serving_default", "--saved_model_tags=serve", str(saved), str(WEBAPP_MODEL)]
    converter.pip_main()
    size = sum(p.stat().st_size for p in WEBAPP_MODEL.iterdir()) / 1e6
    print(f"Exported to {WEBAPP_MODEL} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
