import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from ada_cfar import AdaCFAR1D, get_dataset, parse_tfrecord_fn


def visualize_predictions(model, record_path, num_samples=3):
    print(f"\nExtracting {num_samples} samples for visual inspection...")

    raw_ds = tf.data.TFRecordDataset(record_path)
    ds = raw_ds.map(parse_tfrecord_fn).shuffle(100).batch(1).take(num_samples)

    for i, (profile, true_mask) in enumerate(ds):
        # FIX 2: Use direct tensor invocation instead of model.predict()
        # training=False ensures BatchNormalization uses its locked moving averages
        pred_mask = model(profile, training=False)

        # Convert EagerTensors to numpy arrays and flatten them
        profile_1d = profile.numpy().squeeze()
        true_mask_1d = true_mask.numpy().squeeze()
        pred_mask_1d = pred_mask.numpy().squeeze()

        # Threshold at 0.5 to create the final CFAR detection mask
        pred_binary = (pred_mask_1d > 0.5).astype(float)

        gates = np.arange(len(profile_1d))

        # --- Plotting ---
        fig, ax1 = plt.subplots(figsize=(14, 5))

        color1 = 'tab:blue'
        ax1.set_xlabel('Range Gate Index')
        ax1.set_ylabel('Normalized Amplitude', color=color1)
        ax1.plot(gates, profile_1d, color=color1, alpha=0.7, label='Radar Profile')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()

        ax2.plot(gates, true_mask_1d, color='tab:red', linestyle='--', linewidth=2, label='True Target Mask')
        # Made the predicted mask slightly thicker and semi-transparent so you can see overlap
        ax2.plot(gates, pred_binary, color='tab:green', linewidth=4, alpha=0.5, label='Predicted AdaCFAR Mask')

        ax2.set_ylabel('Detection (0 or 1)')
        ax2.set_ylim(-0.1, 1.1)

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

        plt.title(f"AdaCFAR-1D Detection vs Ground Truth | Sample {i + 1}")
        fig.tight_layout()
        plt.show()


def main():
    val_record_path = "../database/tfrecords/val.tfrecord"
    model_path = "adacfar_best.keras"
    batch_size = 512

    print("Loading AdaCFAR-1D Model...")
    try:
        # FIX 1: compile=False strips the broken optimizer state out during loading
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects={
                'dice_loss': AdaCFAR1D.dice_loss,
                'dice_coef': AdaCFAR1D.dice_coef
            }
        )
    except OSError:
        print(f"Error: Could not find '{model_path}'. Ensure training has finished.")
        return

    # Re-compile the model cleanly just for the evaluation step
    model.compile(
        loss=AdaCFAR1D.dice_loss,
        metrics=[AdaCFAR1D.dice_coef]
    )

    print(f"\n--- Running Full Dataset Evaluation on {val_record_path} ---")
    val_ds = get_dataset(val_record_path, batch_size=batch_size)

    results = model.evaluate(val_ds, verbose=1)

    metrics_dict = dict(zip(model.metrics_names, results))
    print("\nValidation Results:")
    for name, val in metrics_dict.items():
        print(f"  -> {name.upper()}: {val:.4f}")

    # Visual Verification
    visualize_predictions(model, val_record_path, num_samples=3)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # Mixed precision usually isn't strictly necessary for testing, but keeps it consistent
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    main()