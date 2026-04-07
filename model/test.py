import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from database.rad_yolo_dataset import RADYOLODataset
from model.rad_yolo import RADYOLO


def run_snr_sweep(model, dataset_obj, snr_range):
    """Measures Coordinate Error (RMSE) across different noise levels."""
    rmse_results = []

    for snr in snr_range:
        errors = []
        for _ in range(50):  # 50 samples per SNR level
            img, label = dataset_obj.get_sample(has_target=True, snr_db=snr)
            pred = model.predict(img[np.newaxis, ...], verbose=0)[0]

            # Find strongest detection
            idx = np.unravel_index(np.argmax(pred[..., 0]), (16, 16))

            # True vs Pred (Range/Doppler indices)
            true_idx = np.argwhere(label[..., 0] == 1)[0]
            true_pos = true_idx + label[true_idx[0], true_idx[1], 1:]
            pred_pos = np.array(idx) + pred[idx[0], idx[1], 1:]

            err = np.sqrt(np.mean((true_pos - pred_pos) ** 2))
            errors.append(err)

        rmse_results.append(np.mean(errors))
        print(f"SNR: {snr}dB | Average RMSE: {rmse_results[-1]:.4f}")

    return rmse_results


def main():
    config_path = "../database/configs"
    model_path = "rad_yolo_best.keras"

    # Load model with custom loss
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'custom_loss': RADYOLO.custom_loss}
    )

    dataset = RADYOLODataset(config_path)

    # 1. Visualization at 15dB
    print("Visualizing detection at 15dB...")
    net = RADYOLO()
    net.model = model  # Inject loaded weights
    net.test_and_visualize(dataset, snr_db=15)

    # 2. The Stress Test: SNR Sweep
    print("\nStarting SNR Stress Test...")
    snr_levels = [25, 20, 15, 10, 5, 0, -5]
    results = run_snr_sweep(model, dataset, snr_levels)

    # Plotting Performance Degradation
    plt.figure(figsize=(8, 5))
    plt.plot(snr_levels, results, marker='o', linestyle='-', color='red')
    plt.gca().invert_xaxis()
    plt.title("RAD-YOLO Localization Precision vs. SNR")
    plt.xlabel("SNR (dB)")
    plt.ylabel("RMSE (Grid Cell Units)")
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()