"""
train_currency_model.py — VisPay currency recognition model training.

Converted from mlproject.ipynb. Run this in VS Code or any terminal.

Usage:
    python train_currency_model.py

Requirements:
    pip install tensorflow scikit-learn matplotlib numpy pillow

Outputs (all written to OUTPUT_DIR):
    VisPay_currency_model.tflite   — edge inference model (use this in infer.py)
    VisPay_currency_model.h5       — full Keras model (for further fine-tuning)
    class_map.json                 — class index → label map (read by infer.py)
    confusion_matrix.png           — test set confusion matrix
    training_curves.png            — accuracy + loss curves
    batch_test_results/            — misclassified images

Configuration:
    Edit the PATHS and SETTINGS sections below to match your local dataset.
"""

import os
import sys
import json
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
# Change these to match where you extracted the Kaggle dataset
DATASET_ROOT = r"C:\projects\vispay\data\Indian currency dataset v1"
TRAIN_DIR    = os.path.join(DATASET_ROOT, "training")
VAL_DIR      = os.path.join(DATASET_ROOT, "validation")
TEST_DIR     = os.path.join(DATASET_ROOT, "test")

# Where to save model files and results
OUTPUT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "currency_recognition", "models", "currency")
RESULTS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "currency_recognition", "train_results")

# ── Settings ──────────────────────────────────────────────────────────────────
IMAGE_SIZE    = (224, 224)
BATCH_SIZE    = 32
EPOCHS_PHASE1 = 20      # frozen base
EPOCHS_PHASE2 = 10      # fine-tune last 50 layers
SEED          = 865

CLASSES = ["10", "100", "20", "200", "2000", "50", "500", "Background"]


# ── Setup ─────────────────────────────────────────────────────────────────────

def setup_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Output  : {OUTPUT_DIR}")
    print(f"Results : {RESULTS_DIR}")


def check_dataset():
    """Verify dataset folders exist and print class counts."""
    ok = True
    for split, path in [("train", TRAIN_DIR), ("val", VAL_DIR), ("test", TEST_DIR)]:
        if not os.path.exists(path):
            print(f"[ERROR] {split} dir not found: {path}")
            ok = False
        else:
            classes_found = [d for d in os.listdir(path)
                             if os.path.isdir(os.path.join(path, d))]
            total = sum(len(os.listdir(os.path.join(path, c)))
                        for c in classes_found)
            print(f"  {split:6s}: {len(classes_found)} classes, {total} images — {path}")
    if not ok:
        print("\nFix dataset paths in the PATHS section at the top of this file.")
        sys.exit(1)


# ── Data loaders ──────────────────────────────────────────────────────────────

def build_data_generators():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.05,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=False,   # notes have fixed orientation
        fill_mode="nearest",
    )
    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_data = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", shuffle=True, seed=SEED,
    )
    val_data = val_test_datagen.flow_from_directory(
        VAL_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", shuffle=False,
    )
    test_data = val_test_datagen.flow_from_directory(
        TEST_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", shuffle=False,
    )

    # Save class map so infer.py always stays in sync with training
    class_map = {str(v): k for k, v in train_data.class_indices.items()}
    map_path = os.path.join(OUTPUT_DIR, "class_map.json")
    with open(map_path, "w") as f:
        json.dump(class_map, f, indent=2)
    print(f"Class map saved → {map_path}")
    print(f"Class order: {train_data.class_indices}")

    return train_data, val_data, test_data


# ── Class weights ─────────────────────────────────────────────────────────────

def compute_weights(train_data):
    from sklearn.utils.class_weight import compute_class_weight
    cw = compute_class_weight(
        "balanced",
        classes=np.unique(train_data.classes),
        y=train_data.classes,
    )
    weights = dict(enumerate(cw))
    print("Class weights:", {CLASSES[k]: round(v, 2) for k, v in weights.items()})
    return weights


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes: int = 8):
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

    base = MobileNetV2(input_shape=(*IMAGE_SIZE, 3), include_top=False,
                       weights="imagenet")
    base.trainable = False

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=output)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"Model built — {model.count_params():,} params "
          f"({sum(p.numpy().size for p in model.trainable_variables):,} trainable)")
    return model, base


# ── Training ──────────────────────────────────────────────────────────────────

def train_phase1(model, train_data, val_data, class_weights):
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    print("\n── Phase 1: frozen base ──────────────────────────────")
    early_stop = EarlyStopping(monitor="val_loss", patience=5,
                               restore_best_weights=True)
    reduce_lr  = ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                   patience=3, min_lr=1e-6)

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS_PHASE1,
        class_weight=class_weights,
        callbacks=[early_stop, reduce_lr],
    )
    return history


def train_phase2(model, base_model, train_data, val_data, class_weights):
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    print("\n── Phase 2: fine-tune last 50 layers ────────────────")
    base_model.trainable = True
    for layer in base_model.layers[:-50]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Fresh callback instances — reusing phase 1 instances causes early stop
    # to fire immediately because internal best_val_loss state carries over
    early_stop_ft = EarlyStopping(monitor="val_loss", patience=3,
                                  restore_best_weights=True)
    reduce_lr_ft  = ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                      patience=2, min_lr=1e-7)

    history_ft = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS_PHASE2,
        class_weight=class_weights,
        callbacks=[early_stop_ft, reduce_lr_ft],
    )
    return history_ft


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_tflite(tflite_path: str, test_dir: str):
    """Run TFLite inference on every test image and report metrics."""
    import tensorflow as tf
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    from sklearn.metrics import classification_report
    from tensorflow.keras.preprocessing import image as keras_image

    print(f"\n── TFLite evaluation ─────────────────────────────────")
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp  = interp.get_input_details()
    outp = interp.get_output_details()

    class_names = sorted(os.listdir(test_dir))
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    y_true, y_pred, times = [], [], []
    save_dir = os.path.join(RESULTS_DIR, "misclassified")
    os.makedirs(save_dir, exist_ok=True)

    for cls in class_names:
        cls_path = os.path.join(test_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        for fname in os.listdir(cls_path):
            img_path = os.path.join(cls_path, fname)
            try:
                img = keras_image.load_img(img_path, target_size=IMAGE_SIZE)
                tensor = np.expand_dims(
                    keras_image.img_to_array(img).astype(np.float32) / 255.0,
                    axis=0,
                )
                t0 = time.time()
                interp.set_tensor(inp[0]["index"], tensor)
                interp.invoke()
                out = interp.get_tensor(outp[0]["index"])[0]
                times.append((time.time() - t0) * 1000)

                pred = int(np.argmax(out))
                true = class_to_idx[cls]
                y_true.append(true)
                y_pred.append(pred)

                if pred != true:
                    dest = os.path.join(
                        save_dir,
                        f"{fname}_true_{cls}_pred_{class_names[pred]}.png",
                    )
                    plt.imsave(dest, tensor[0])
            except Exception as e:
                print(f"  Skipping {img_path}: {e}")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Per-class report — more useful than overall accuracy for a payment app
    print("\nPer-class report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    print(f"Inference  avg: {np.mean(times):.1f} ms  "
          f"worst: {np.max(times):.1f} ms  "
          f"best: {np.min(times):.1f} ms")

    # Confusion matrix
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(cmap="Blues", ax=ax, values_format="d")
    ax.set_title("TFLite test set confusion matrix")
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved → {cm_path}")
    print(f"Misclassified images   → {save_dir}")


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_training(h1, h2=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    acc  = h1.history["accuracy"]
    val  = h1.history["val_accuracy"]
    loss = h1.history["loss"]
    vloss= h1.history["val_loss"]

    if h2:
        acc   += h2.history["accuracy"]
        val   += h2.history["val_accuracy"]
        loss  += h2.history["loss"]
        vloss += h2.history["val_loss"]
        split  = len(h1.history["accuracy"])
        axes[0].axvline(split - 1, color="gray", linestyle="--", alpha=0.5,
                        label="fine-tune start")
        axes[1].axvline(split - 1, color="gray", linestyle="--", alpha=0.5,
                        label="fine-tune start")

    axes[0].plot(acc,  label="train acc")
    axes[0].plot(val,  label="val acc")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(loss,  label="train loss")
    axes[1].plot(vloss, label="val loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "training_curves.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved → {path}")


# ── TFLite export ─────────────────────────────────────────────────────────────

def export_tflite(model, train_data):
    import tensorflow as tf

    # Standard float32 export
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_path = os.path.join(OUTPUT_DIR, "VisPay_currency_model.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite (float32) saved → {tflite_path}  "
          f"({os.path.getsize(tflite_path)//1024} KB)")

    # INT8 quantised export (smaller + faster on CPU)
    def representative_data():
        gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)
        data = gen.flow_from_directory(
            TRAIN_DIR, target_size=IMAGE_SIZE, batch_size=1,
            class_mode=None, shuffle=True, seed=42,
        )
        for i, img in enumerate(data):
            if i >= 200:
                break
            yield [img.astype(np.float32)]

    conv_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
    conv_int8.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
    conv_int8.representative_dataset = representative_data
    conv_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv_int8.inference_input_type  = tf.float32
    conv_int8.inference_output_type = tf.float32

    try:
        tflite_int8 = conv_int8.convert()
        int8_path = os.path.join(OUTPUT_DIR, "VisPay_currency_model_int8.tflite")
        with open(int8_path, "wb") as f:
            f.write(tflite_int8)
        print(f"TFLite (int8)    saved → {int8_path}  "
              f"({os.path.getsize(int8_path)//1024} KB)")
    except Exception as e:
        print(f"INT8 export failed (non-critical): {e}")

    return tflite_path


def export_h5(model):
    h5_path = os.path.join(OUTPUT_DIR, "VisPay_currency_model.h5")
    model.save(h5_path)
    print(f"Keras .h5        saved → {h5_path}  "
          f"({os.path.getsize(h5_path)//1024} KB)")
    return h5_path


# ── Run inference with confidence (was missing from notebook) ─────────────────

def run_inference_with_confidence(interpreter, tensor, threshold=0.60):
    """
    Run TFLite inference and return a result dict.
    Returns label='UNCERTAIN' when max probability < threshold.

    Args:
        interpreter: loaded tf.lite.Interpreter
        tensor:      (1, 224, 224, 3) float32 numpy array
        threshold:   minimum probability to commit to a class

    Returns:
        {"label": int or "UNCERTAIN", "prob": float, "all_probs": array}
    """
    inp  = interpreter.get_input_details()
    outp = interpreter.get_output_details()

    interpreter.set_tensor(inp[0]["index"], tensor)
    interpreter.invoke()
    out = interpreter.get_tensor(outp[0]["index"])[0]

    class_id = int(np.argmax(out))
    prob     = float(out[class_id])

    if prob < threshold:
        return {"label": "UNCERTAIN", "prob": prob, "all_probs": out}
    return {"label": class_id, "prob": prob, "all_probs": out}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 58)
    print("  VisPay currency model training")
    print("=" * 58)

    setup_dirs()
    check_dataset()

    print("\nBuilding data generators...")
    train_data, val_data, test_data = build_data_generators()
    class_weights = compute_weights(train_data)

    print("\nBuilding model...")
    model, base_model = build_model(num_classes=len(train_data.class_indices))

    print("\nStarting training...")
    h1 = train_phase1(model, train_data, val_data, class_weights)
    h2 = train_phase2(model, base_model, train_data, val_data, class_weights)

    print("\nSaving models...")
    tflite_path = export_tflite(model, train_data)
    export_h5(model)

    print("\nPlotting training curves...")
    plot_training(h1, h2)

    print("\nEvaluating TFLite model on test set...")
    evaluate_tflite(tflite_path, TEST_DIR)

    print("\n" + "=" * 58)
    print("  Training complete.")
    print(f"  Model : {tflite_path}")
    print(f"  Results: {RESULTS_DIR}")
    print("=" * 58)


if __name__ == "__main__":
    main()