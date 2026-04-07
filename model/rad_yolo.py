import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers
import matplotlib.pyplot as plt
import numpy as np


class RADYOLO:
    def __init__(self, input_shape=(256, 256, 1), grid_size=16):
        self.input_shape = input_shape
        self.grid_size = grid_size
        self.model = self._build_model()

    def _build_model(self):
        """Builds the model using the Sequential API with named layers."""
        model = models.Sequential([
            layers.Input(shape=self.input_shape, name="Input_RD_Map"),

            # Backbone - Feature Extraction
            layers.Conv2D(32, (3, 3), padding='same', activation='relu', name="Backbone_Conv_1"),
            layers.BatchNormalization(name="BN_1"),
            layers.MaxPooling2D((2, 2), name="Pool_1"),  # 256 -> 128

            layers.Conv2D(64, (3, 3), padding='same', activation='relu', name="Backbone_Conv_2"),
            layers.BatchNormalization(name="BN_2"),
            layers.MaxPooling2D((2, 2), name="Pool_2"),  # 128 -> 64

            layers.Conv2D(128, (3, 3), padding='same', activation='relu', name="Backbone_Conv_3"),
            layers.BatchNormalization(name="BN_3"),
            layers.MaxPooling2D((2, 2), name="Pool_3"),  # 64 -> 32

            layers.Conv2D(256, (3, 3), padding='same', activation='relu', name="Backbone_Conv_4"),
            layers.BatchNormalization(name="BN_4"),
            layers.MaxPooling2D((2, 2), name="Pool_4"),  # 32 -> 16

            # Detection Head
            # Output: 16x16 grid with [Conf, delta_r, delta_v]
            layers.Conv2D(3, (1, 1), activation='sigmoid', name="YOLO_Head")
        ], name="RAD_YOLO_Model")

        return model

    @staticmethod
    @tf.function
    def custom_loss(y_true, y_pred):
        # 1. Expand dims for BCE
        obj_true_exp = tf.expand_dims(y_true[..., 0], axis=-1)
        obj_pred_exp = tf.expand_dims(y_pred[..., 0], axis=-1)

        # 2. Weighted Confidence Loss
        # We want to penalize false positives less than true positive misses
        bce = losses.binary_crossentropy(obj_true_exp, obj_pred_exp)
        obj_mask = y_true[..., 0]
        noobj_mask = 1.0 - obj_mask

        # Apply a 0.5 weight to empty cells, and 1.0 to the cell with the target
        conf_weight = (noobj_mask * 0.5) + (obj_mask * 1.0)
        conf_loss = bce * conf_weight

        # 3. Localization Loss (MSE)
        dr_true, dv_true = y_true[..., 1], y_true[..., 2]
        dr_pred, dv_pred = y_pred[..., 1], y_pred[..., 2]

        loc_loss = obj_mask * (tf.square(dr_true - dr_pred) + tf.square(dv_true - dv_pred))

        # MULTIPLY loc_loss by a large factor (e.g., 5.0) to force coordinate precision
        loc_loss = loc_loss * 5.0

        return tf.reduce_mean(conf_loss + loc_loss)

    def compile_model(self, lr=0.001):
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=lr),
            loss=self.custom_loss,
            metrics=[self.objectness_accuracy, self.localization_rmse]
        )

    def train(self, train_gen, val_gen, epochs=20, steps_per_epoch=100, callbacks=[]):
        """Wraps the Keras fit method."""
        return self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=steps_per_epoch // 5,
            callbacks=callbacks
        )

    def test_and_visualize(self, dataset_obj, snr_db=15):
        """Inference and plotting for a single sample."""
        img, label = dataset_obj.get_sample(has_target=True, snr_db=snr_db)

        # Add batch dimension for prediction: (1, 256, 256, 1)
        pred = self.model.predict(img[np.newaxis, ...])[0]

        # Find the cell with the highest confidence in the 16x16 grid
        idx = np.unravel_index(np.argmax(pred[..., 0]), (self.grid_size, self.grid_size))
        conf, dr, dv = pred[idx]

        # Reconstruct pixel coordinates
        cell_size = self.input_shape[0] / self.grid_size
        p_r = (idx[0] + dr) * cell_size
        p_v = (idx[1] + dv) * cell_size

        # Plotting
        plt.imshow(img.squeeze(), origin='lower', cmap='viridis')
        plt.scatter(p_v, p_r, edgecolors='red', facecolors='none', s=100, label=f'Pred (Conf: {conf:.2f})')
        plt.title(f"Inference at {snr_db}dB")
        plt.legend()
        plt.show()

    @staticmethod
    def objectness_accuracy(y_true, y_pred):
        """Measures if the highest confidence cell matches the true target cell."""
        # Get the 16x16 confidence grids
        conf_true = y_true[..., 0]
        conf_pred = y_pred[..., 0]

        # Flatten the spatial dimensions to [Batch, 256]
        conf_true_flat = tf.keras.layers.Flatten()(conf_true)
        conf_pred_flat = tf.keras.layers.Flatten()(conf_pred)

        # Find the index of the cell with the highest confidence
        idx_true = tf.argmax(conf_true_flat, axis=1)
        idx_pred = tf.argmax(conf_pred_flat, axis=1)

        # Compare if the model picked the right cell
        return tf.cast(tf.equal(idx_true, idx_pred), tf.float32)

    @staticmethod
    def localization_rmse(y_true, y_pred):
        """Measures the Root Mean Square Error of the coordinates."""
        obj_mask = y_true[..., 0]

        dr_true, dv_true = y_true[..., 1], y_true[..., 2]
        dr_pred, dv_pred = y_pred[..., 1], y_pred[..., 2]

        mse = obj_mask * (tf.square(dr_true - dr_pred) + tf.square(dv_true - dv_pred))
        # Sum over the grid, take mean over batch, then sqrt
        return tf.sqrt(tf.reduce_mean(tf.reduce_sum(mse, axis=[1, 2])))