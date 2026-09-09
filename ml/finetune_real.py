"""Fine-tunes the existing classifier on synthetic data + a small set of REAL labeled examples
extracted from an actual annotated worksheet photo (real_labels.npz: pos = genuine handwriting/
annotation crops, neg = genuine printed-text crops, both from the same real photo). Real
examples are oversampled since there are only a couple dozen of them against thousands of
synthetic ones — otherwise their gradient contribution would be negligible.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras

IMG_H, IMG_W = 64, 128
REAL_OVERSAMPLE = 25


def load_synthetic(name):
    data = np.load(f"{name}.npz")
    return data["images"].astype("float32") / 255.0, data["labels"].astype("float32")


def load_real():
    data = np.load("real_labels.npz")
    pos = data["pos"].astype("float32") / 255.0
    neg = data["neg"].astype("float32") / 255.0
    return pos, neg


def main():
    train_x, train_y = load_synthetic("train")
    val_x, val_y = load_synthetic("val")
    test_x, test_y = load_synthetic("test")

    real_pos, real_neg = load_real()
    print(f"real: {len(real_pos)} positive, {len(real_neg)} negative")

    # Oversample pos and neg to roughly the SAME final count -- the earlier attempt oversampled
    # both by the same factor (25x), which kept the real class imbalance (14 pos : 221 neg) and
    # made the model dramatically more conservative (31 -> 7 flagged on the same real photo,
    # dropping several previously-correct detections). Balance them against each other instead.
    target_per_class = len(real_pos) * REAL_OVERSAMPLE
    pos_reps = REAL_OVERSAMPLE
    neg_reps = max(1, round(target_per_class / len(real_neg)))
    real_x_over = np.concatenate([np.repeat(real_pos, pos_reps, axis=0), np.repeat(real_neg, neg_reps, axis=0)], axis=0)
    real_y_over = np.concatenate([np.ones(len(real_pos) * pos_reps), np.zeros(len(real_neg) * neg_reps)]).astype("float32")
    print(f"real oversampled: {len(real_pos) * pos_reps} positive, {len(real_neg) * neg_reps} negative")

    combined_x = np.concatenate([train_x, real_x_over], axis=0)
    combined_y = np.concatenate([train_y, real_y_over], axis=0)
    perm = np.random.permutation(len(combined_y))
    combined_x, combined_y = combined_x[perm], combined_y[perm]
    print(f"fine-tune train set: {combined_x.shape}")

    model = keras.models.load_model("handwriting_classifier.keras")
    model.compile(
        optimizer=keras.optimizers.Adam(2e-4),  # lower LR than train.py's 1e-3 -- fine-tuning, not training from scratch
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
    ]

    model.fit(
        combined_x, combined_y,
        validation_data=(val_x, val_y),
        epochs=12,
        batch_size=64,
        callbacks=callbacks,
        verbose=2,
    )

    print("\nEvaluating on held-out SYNTHETIC test set (checking no regression)...")
    results = model.evaluate(test_x, test_y, verbose=0, return_dict=True)
    for k, v in results.items():
        print(f"  test {k}: {v:.4f}")

    print("\nEvaluating on the REAL labeled examples (in-sample, expect near-perfect)...")
    real_x_eval = np.concatenate([real_pos, real_neg], axis=0)
    real_y_eval = np.concatenate([np.ones(len(real_pos)), np.zeros(len(real_neg))]).astype("float32")
    real_results = model.evaluate(real_x_eval, real_y_eval, verbose=0, return_dict=True)
    for k, v in real_results.items():
        print(f"  real {k}: {v:.4f}")

    model.save("handwriting_classifier_finetuned.keras")
    print("\nSaved handwriting_classifier_finetuned.keras")


if __name__ == "__main__":
    main()
