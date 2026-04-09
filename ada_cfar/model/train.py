import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

# Enable Mixed Precision to double throughput on modern GPUs
tf.keras.mixed_precision.set_global_policy('mixed_float16')

from ada_cfar import AdaCFAR1D, get_dataset


def main():
    # 1. Configs
    train_record_path = "../database/tfrecords/train.tfrecord"
    val_record_path = "../database/tfrecords/val.tfrecord"
    model_path = "adacfar_best_v11.keras"

    # Massive batch size because 1D data is incredibly lightweight
    batch_size = 256
    epochs = 300

    # 2. Build High-Speed DMA Datasets
    print("Initializing DMA TFRecord streams...")
    # Add shuffle and repeat to training
    train_ds = get_dataset(train_record_path, batch_size=batch_size).shuffle(1000).repeat()
    # Add repeat to validation
    val_ds = get_dataset(val_record_path, batch_size=batch_size).repeat()

    # Calculate steps based on file size and batch size
    train_steps = 25000 // batch_size  # ~19 steps
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
    net.model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1
    )

    print("Training Complete. Model saved!")


if __name__ == "__main__":
    # Suppress startup warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main()