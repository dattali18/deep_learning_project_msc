import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from database.deep_cfar_dataset import DeepCFARDataset
from model.deep_cfar import DeepCFAR


def compute_iou(y_true, y_pred, threshold=0.5):
    """Calculates Intersection over Union safely handling empty masks."""
    y_true_bin = (y_true > 0.5).astype(bool)
    y_pred_bin = (y_pred > threshold).astype(bool)

    intersection = np.logical_and(y_true_bin, y_pred_bin).sum()
    union = np.logical_or(y_true_bin, y_pred_bin).sum()

    # If there are no targets in reality AND none predicted, it's a perfect match
    if union == 0:
        return 1.0 if np.sum(y_pred_bin) == 0 else 0.0

    return intersection / union


def visualize_prediction(model, ds, snr_db, num_targets=2):
    """Plots Input vs Ground Truth vs Prediction for visual inspection."""
    img, true_mask = ds.get_sample(num_targets=num_targets, snr_db=snr_db)

    # Predict (Add batch dimension, then remove it)
    pred_mask = model.predict(img[np.newaxis, ...], verbose=0)[0]

    # Threshold the prediction for visualization
    pred_mask_bin = (pred_mask > 0.5).astype(float)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Noisy Input
    im0 = axes[0].imshow(img.squeeze(), origin='lower', cmap='viridis')
    axes[0].set_title(f"Input: Noisy RD Map ({snr_db}dB SNR)")
    fig.colorbar(im0, ax=axes[0])

    # 2. Ground Truth Mask
    im1 = axes[1].imshow(true_mask.squeeze(), origin='lower', cmap='gray')
    axes[1].set_title("Ground Truth Mask")
    fig.colorbar(im1, ax=axes[1])

    # 3. Predicted Mask
    im2 = axes[2].imshow(pred_mask_bin.squeeze(), origin='lower', cmap='gray')
    axes[2].set_title(f"Predicted Mask (IoU: {compute_iou(true_mask, pred_mask):.2f})")
    fig.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.show()


def run_snr_sweep(model, ds, snr_range, samples_per_snr=30):
    """Tests the model across degrading noise floors."""
    iou_results = []

    for snr in snr_range:
        ious = []
        for _ in range(samples_per_snr):
            # Randomize targets to test both detection and false alarm suppression
            num_targets = np.random.randint(0, 4)
            img, true_mask = ds.get_sample(num_targets=num_targets, snr_db=snr)

            pred_mask = model.predict(img[np.newaxis, ...], verbose=0)[0]
            ious.append(compute_iou(true_mask, pred_mask))

        avg_iou = np.mean(ious)
        iou_results.append(avg_iou)
        print(f"SNR: {snr:3}dB | Average IoU: {avg_iou:.4f}")

    return iou_results


def main():
    config_path = "../database/configs"
    # Note: Using the filename from the previous train.py script
    model_path = "deep_cfar_best.keras"

    print("Loading Model...")
    try:
        # Pass the custom loss and metric to the Keras loader
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'dice_loss': DeepCFAR.dice_loss,
                'dice_coef': DeepCFAR.dice_coef
            }
        )
    except OSError:
        print(f"Error: Could not find '{model_path}'. Did the training script finish?")
        return

    ds = DeepCFARDataset(config_path)

    # 1. Visual Verification at high SNR (Sanity Check)
    print("\n--- Visual Verification (20dB) ---")
    visualize_prediction(model, ds, snr_db=20, num_targets=2)

    # 2. Visual Verification at low SNR (Stress Check)
    print("\n--- Visual Verification (5dB) ---")
    visualize_prediction(model, ds, snr_db=5, num_targets=2)

    # 3. The SNR Degradation Sweep
    print("\n--- Starting SNR Stress Test ---")
    snr_levels = [25, 20, 15, 10, 5, 0, -5]
    iou_scores = run_snr_sweep(model, ds, snr_levels)

    # Plotting the degradation curve
    plt.figure(figsize=(8, 5))
    plt.plot(snr_levels, iou_scores, marker='o', linestyle='-', color='blue', linewidth=2)
    plt.gca().invert_xaxis()  # Reverse X-axis so it drops from high SNR to low SNR
    plt.title("Deep CFAR-Net Segmentation Accuracy vs. SNR")
    plt.xlabel("Signal-to-Noise Ratio (dB)")
    plt.ylabel("Intersection over Union (IoU)")
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Failure Threshold')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()