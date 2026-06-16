import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Enable Mixed Precision to double throughput on modern GPUs
tf.keras.mixed_precision.set_global_policy('mixed_float16')

from ada_cfar import AdaCFAR1D, get_dataset

def plot_training_history(history, output_dir: str = "training_plots"):
    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------
    # 1. Loss plot: train vs validation
    # --------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "train_val_loss.png"), dpi=300)
    plt.close()

    # --------------------------------------------------
    # 2. Dice coefficient plot: train vs validation
    # --------------------------------------------------
    if "dice_coef" in history.history and "val_dice_coef" in history.history:
        plt.figure(figsize=(8, 5))
        plt.plot(history.history["dice_coef"], label="Train Dice")
        plt.plot(history.history["val_dice_coef"], label="Validation Dice")
        plt.xlabel("Epoch")
        plt.ylabel("Dice coefficient")
        plt.title("Training and Validation Dice Coefficient")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "train_val_dice.png"), dpi=300)
        plt.close()

    # --------------------------------------------------
    # 3. Save raw history for report/presentation reuse
    # --------------------------------------------------
    history_path = os.path.join(output_dir, "training_history.csv")

    keys = list(history.history.keys())
    num_epochs = len(history.history[keys[0]])

    with open(history_path, "w", encoding="utf-8") as f:
        f.write("epoch," + ",".join(keys) + "\n")

        for epoch_idx in range(num_epochs):
            values = [str(epoch_idx + 1)]
            for key in keys:
                values.append(str(history.history[key][epoch_idx]))
            f.write(",".join(values) + "\n")


def main():
    # 1. Configs
    train_record_path = "../database/tfrecords/train.tfrecord"
    val_record_path = "../database/tfrecords/val.tfrecord"
    model_path = "adacfar_best_v05.keras"

    # Massive batch size because 1D data is incredibly lightweight
    batch_size = 256
    epochs = 300

    # 2. Build High-Speed DMA Datasets
    print("Initializing DMA TFRecord streams...")
    # Add shuffle and repeat to training
    train_ds = get_dataset(train_record_path, batch_size=batch_size).shuffle(100).repeat()
    # Add repeat to validation
    val_ds = get_dataset(val_record_path, batch_size=batch_size).repeat()

    # Calculate steps based on file size and batch size
    train_steps = 2500 // batch_size  # ~19 steps
    val_steps = 1000 // batch_size  # ~1 step

    # 3. Initialize and Compile
    print("Building AdaCFAR-1D Model...")
    net = AdaCFAR1D(version=2)
    net.compile_model(lr=0.001)

    # Print the model summary to see how few parameters it actually has!
    net.model.summary()

    # 4. Callbacks
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True),
        # TensorBoard(log_dir="./logs/fit", histogram_freq=1)
    ]

    # 5. Training
    print("\n--- Starting High-Speed Training ---")
    history = net.model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1
    )

    plot_training_history(history, output_dir="training_plots")

    print("Training Complete. Model saved!")


if __name__ == "__main__":
    # Suppress startup warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main()