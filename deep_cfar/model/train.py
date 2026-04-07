import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from database.deep_cfar_dataset import DeepCFARDataset, data_generator
from model.deep_cfar import DeepCFAR

def plot_training_history(history):
    """
    Plots the training and validation accuracy and loss side-by-side.
    Accepts the history object returned by model.fit().
    """
    # Extract data from the history dictionary
    # Keras sometimes uses 'acc' vs 'accuracy' depending on the version
    acc = history.history.get('accuracy', history.history.get('acc', []))
    val_acc = history.history.get('val_accuracy', history.history.get('val_acc', []))
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(loss) + 1)

    # Create a figure with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Plot 1: Accuracy ---
    if acc and val_acc:
        ax1.plot(epochs, acc, 'b-', label='Training Accuracy', linewidth=2)
        ax1.plot(epochs, val_acc, 'r--', label='Validation Accuracy', linewidth=2)
        ax1.set_title('Training and Validation Accuracy')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        ax1.set_title('Accuracy')
        ax1.text(0.5, 0.5, 'Accuracy metric not found in history.\nEnsure metrics=["accuracy"] is in model.compile()',
                 ha='center', va='center', transform=ax1.transAxes)

    # --- Plot 2: Loss ---
    ax2.plot(epochs, loss, 'b-', label='Training Loss', linewidth=2)
    ax2.plot(epochs, val_loss, 'r--', label='Validation Loss', linewidth=2)
    ax2.set_title('Training and Validation Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Adjust layout and display
    plt.tight_layout()
    plt.show()


def main():
    # 1. Configs and Hyperparameters
    config_path = "../database/configs"
    model_path = "deep_cfar_best.keras"
    batch_size = 32

    # Increased total epochs and defined steps explicitly
    epochs = 30
    train_steps = 100

    # 2. Setup Data Factory (Instantiate ONCE)
    ds = DeepCFARDataset(config_path)

    output_signature = (
        tf.TensorSpec(shape=(256, 256, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(256, 256, 1), dtype=tf.float32)  # Note: Label is now 256x256x1
    )

    # Use a lambda to pass the single 'ds' instance into the generator
    train_ds = tf.data.Dataset.from_generator(
        lambda: data_generator(ds),
        output_signature=output_signature
    ).batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_generator(
        lambda: data_generator(ds),
        output_signature=output_signature
    ).batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)

    # 3. Initialize and Compile RAD-YOLO
    net = DeepCFAR()
    net.compile_model(lr=0.001)

    # 4. Callbacks
    callbacks = [
        # Increased patience to 12 to ride out noisy validation spikes
        ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True),
        EarlyStopping(
            monitor='val_loss',
            patience=12,
            restore_best_weights=True,
            verbose=1
        ),
    ]

    # 5. Training
    print("Starting RAD-YOLO Training...")
    history = net.train(
        train_ds,
        val_gen=val_ds,
        epochs=epochs,
        steps_per_epoch=train_steps,
        callbacks=callbacks
    )

    # Save final weights just in case
    net.model.save("deep_cfar_final.keras")
    print("Training Complete. Model saved to deep_cfar_final.keras")

    # Plot the history
    plot_training_history(history)

if __name__ == "__main__":
    main()
