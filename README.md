# কৃষি ডাক্তার · Krishi Daktar (Bangla Crop Doctor)

A farmer takes or uploads a leaf photo. A computer-vision model **running fully inside the phone's browser** (no server, works offline) names the likely disease in **Bangla + English**, shows confidence, the cause and 4 simple tips. Built for the Craft N Code West Bengal hackathon (AgriTech track).

**Live demo:** https://melodious-moxie-41dcf7.netlify.app  (works offline after the first load)

## Problem
Small farmers in West Bengal often notice crop disease late and cannot reach an agriculture officer quickly. Wrong or delayed treatment costs yield. Many have poor internet and prefer Bangla.

## Approach
| Part | Choice |
|---|---|
| Crops | Potato, tomato, rice, sugarcane (21 classes: healthy + main diseases) |
| Model | MobileNetV2 (ImageNet) transfer learning, Keras 3 / TensorFlow |
| Training | 2 phases: train new head, then fine-tune top 50 layers |
| Augmentation | flip, rotation, zoom, shift, brightness, contrast, blur, **random background replacement** (lab images have plain backgrounds) |
| Imbalance | per-sample class weights |
| Export | TensorFlow.js `tfjs_graph_model` (about 9 MB) |
| App | Plain HTML/CSS/JS, TF.js in browser, PWA (service worker + manifest), Bangla default + English toggle |
| QR camera (computer -> phone) | On a computer, "Take photo" shows a QR code. The phone scans it, takes the photo and uploads it over HTTPS to the free public relay ntfy.sh under a random unguessable topic (ntfy deletes it after about 3 hours); the computer picks it up and the diagnosis runs in the computer's browser. Needs internet on both devices. |
| Hybrid online AI (optional) | Offline: the on-device model above (potato, tomato, rice). Online: the photo is also sent to our own Netlify Function `/api/diagnose`, which asks a vision AI (Google Gemini). It knows many more crops and diseases (and rice leaf blast / healthy rice) and writes simple Bangla advice. If both agree the app shows "verified"; if they differ it shows the more confident one plus a warning and the other answer. Any online failure silently falls back to the on-device result. The user can switch the online check off. Needs the environment variable `GEMINI_API_KEY` in Netlify (never in the page). |
| Safety | Below **70%** confidence the app says "I am not sure" and sends the farmer to the agriculture officer (Krishi Bibhag). Top-2 predictions are always shown. Disclaimer always visible. |

Classes: potato (early blight, late blight, healthy); tomato (bacterial spot, early blight, late blight, leaf mold, Septoria leaf spot, spider mites, target spot, mosaic virus, yellow leaf curl virus, healthy); rice (bacterial leaf blight, brown spot, leaf smut).

### Online AI setup
Netlify -> Site configuration -> Environment variables -> add `GEMINI_API_KEY` from https://aistudio.google.com/apikey (and optionally `GEMINI_MODEL`, default `gemini-2.5-flash`), set a spend limit on the key, then redeploy. Photos are sent only when online and the switch is on.

## App features (mobile-app style)
Home (live weather + disease-risk hints, Open-Meteo), Scan (on-device model, optional online AI, photo-quality check, listen-aloud and share), Crop guide (offline), AI helper chat (Bangla, voice input, knows the last scan and the weather), My farm (scan history stored on the device, settings, install as app). Online AI answers add severity, spread risk, prevention and a follow-up check question. No account or backend database: history never leaves the device.

## Project structure
```
training/      config.py, prepare_data.py, train.py, evaluate.py, export_tfjs.py, krishi_daktar_colab.ipynb
webapp/        index.html, style.css, app.js, i18n.js, sw.js, manifest.webmanifest, icons/, netlify.toml
webapp/data/   labels.json (class index -> id), diseases.json (Bangla + English cause and tips)
webapp/model/  model.json + weight shards (TF.js)
results/       confusion_matrix.png, per_class_accuracy.png, training_curve.png, metrics.json
```

## Datasets (download yourself, put next to `training/`)
- PlantVillage (potato, tomato folders) -> `PlantVillage/<class folders>`
- Rice leaf disease images (Bacterial leaf blight, Brown spot, Leaf smut) -> `rice_leaf_diseases/<class folders>`

## How to run
**Train (Google Colab recommended, free T4 GPU, about 1 hour):**
1. `python training/prepare_data.py` builds a train/val/test split (70/15/15, 256 px) and `colab_bundle.zip`.
2. Upload `colab_bundle.zip` to Google Drive (My Drive), open `training/krishi_daktar_colab.ipynb` in Colab, run all cells (train, evaluate, export).
3. Copy `webapp/model/` and `results/` back to this folder.

**Run the app locally:**
```
cd webapp
python -m http.server 8080      # open http://localhost:8080
```
Camera and service worker need `localhost` or HTTPS.

**Deploy:** publish the `webapp/` folder on Netlify (`netlify.toml` included).

## Evaluation (held-out test split, 2,334 images, 21-class model)
| Metric | Result |
|---|---|
| Overall top-1 accuracy | **92.6%** |
| Top-2 accuracy | **97.6%** |
| Share of images with confidence >= 70% | 89.8% |
| Accuracy on those confident predictions | **96.7%** |

Per-class accuracy (see `results_v3/per_class_accuracy.png`, `results_v3/confusion_matrix.png`):

| Crop | Class | Acc. | Test n |
|---|---|---|---|
| potato | potato_early_blight | 98.7% | 150 |
| potato | potato_healthy | 100.0% | 24 |
| potato | potato_late_blight | 97.3% | 150 |
| rice | rice_bacterial_leaf_blight | 100.0% | 6 |
| rice | rice_brown_spot | 100.0% | 6 |
| rice | rice_leaf_smut | 66.7% | 6 |
| sugarcane | sugarcane_healthy | 97.5% | 79 |
| sugarcane | sugarcane_mosaic | 91.4% | 70 |
| sugarcane | sugarcane_red_rot | 83.3% | 78 |
| sugarcane | sugarcane_rust | 89.7% | 78 |
| sugarcane | sugarcane_yellow_leaf | 97.4% | 76 |
| tomato | tomato_bacterial_spot | 94.4% | 180 |
| tomato | tomato_early_blight | 72.7% | 150 |
| tomato | tomato_healthy | 99.4% | 180 |
| tomato | tomato_late_blight | 86.7% | 180 |
| tomato | tomato_leaf_mold | 97.2% | 144 |
| tomato | tomato_mosaic_virus | 100.0% | 57 |
| tomato | tomato_septoria_leaf_spot | 94.4% | 180 |
| tomato | tomato_spider_mites | 96.7% | 180 |
| tomato | tomato_target_spot | 80.0% | 180 |
| tomato | tomato_yellow_leaf_curl_virus | 98.9% | 180 |

Adding sugarcane (5 classes) and a shorter training run lowered overall accuracy from 95.1% (16-class model) to the figure above; tomato early blight and target spot are the weakest tomato classes. The earlier 16-class results are kept in `results/`.

**Field-like stress test (informal, 52 images, synthetic):** test leaves were shrunk to 50-80% of the frame and placed on random cluttered backgrounds with blur and brightness changes. A single model pass scored **61.5%**; averaging 4 views in the browser (test-time augmentation: original, mirror, two centre zooms, now used by the app) scored **73.1%**. This confirms the lab-to-field gap: the 95% number above should not be expected on real farm photos. Retraining with field-like augmentation and real field photos is the main next step.

## Honest limitations
- **Lab vs real field.** PlantVillage photos are single leaves on plain backgrounds, taken in controlled light. The 95% above is on the same kind of images. Real field photos (clutter, shadows, several leaves, low-end cameras) will score lower. Augmentation helps but we have **not** measured field accuracy yet.
- **Test split is optimistic.** The split is random per image; PlantVillage has several photos of the same leaf, so some near-duplicates may sit in both train and test.
- **Rice is weak.** Only 40 images per class were available, so only 6 test images per rice class. Those numbers (including 50% for leaf smut) are statistically meaningless. No *healthy rice* class and no *leaf blast* class exist in the data we used. Rice results should be treated as a demo.
- **Tomato early blight (78%)** is often confused with other leaf-spot diseases that look similar.
- A leaf the model was never trained on can still be forced into a wrong class. The 70% threshold reduces this but cannot remove it.
- Advice is general; it deliberately names no pesticide dose. It is **not a professional diagnosis**.

## Next steps
Collect field photos from local farms, add healthy rice and leaf blast, re-split by plant (not by image), add more crops and voice output in Bangla.
