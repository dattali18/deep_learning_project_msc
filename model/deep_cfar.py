import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers
import tensorflow.keras.backend as K

class DeepCFAR:
    def __init__(self, input_shape=(256, 256, 1)):
        self.input_shape = input_shape
        self.model = self._build_model()

    def _build_model(self):
        """Builds the Dilated CFAR Network."""
        model = models.Sequential([
            layers.Input(shape=self.input_shape, name="Input_RD_Map"),

            # 1. Local Block (The "Cell Under Test" equivalent)
            # Standard 3x3 convolution to capture the immediate peak shape.
            layers.Conv2D(32, (3, 3), padding='same', activation='relu', name="Local_CUT_Conv"),
            layers.BatchNormalization(name="BN_Local"),

            # 2. Guard Context Block
            # Dilation of 2 expands the kernel to look slightly past the immediate center.
            layers.Conv2D(32, (3, 3), padding='same', dilation_rate=2, activation='relu', name="Guard_Conv"),
            layers.BatchNormalization(name="BN_Guard"),

            # 3. Training Context Block 1
            # Dilation of 4 reaches further into the noise floor.
            layers.Conv2D(32, (3, 3), padding='same', dilation_rate=4, activation='relu', name="Train_Conv_1"),
            layers.BatchNormalization(name="BN_Train_1"),

            # 4. Training Context Block 2
            # Dilation of 8 provides a massive receptive field to establish the background average.
            layers.Conv2D(32, (3, 3), padding='same', dilation_rate=8, activation='relu', name="Train_Conv_2"),
            layers.BatchNormalization(name="BN_Train_2"),

            # 5. Fusion Block
            # A 1x1 convolution acts as a per-pixel fully connected layer.
            # It mathematically compares the CUT features against the Training features.
            layers.Conv2D(16, (1, 1), padding='same', activation='relu', name="Fusion_1x1"),

            # 6. Output Mask
            # 1 filter with Sigmoid outputs the 256x256 probability mask [0, 1]
            layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid', name="Output_Mask")
        ], name="Dilated_CFAR_Net")

        return model

    @staticmethod
    def dice_coef(y_true, y_pred, smooth=1e-6):
        """
        Calculates the Dice Coefficient.
        Smooth factor prevents division by zero.
        """
        # Flatten the tensors to 1D arrays
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)

        # Calculate intersection
        intersection = K.sum(y_true_f * y_pred_f)

        # Formula: (2 * Intersection) / (Sum of pixels in True + Sum of pixels in Pred)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    @staticmethod
    def dice_loss(y_true, y_pred):
        """
        Loss function that directly minimizes the Dice score.
        """
        return 1.0 - DeepCFAR.dice_coef(y_true, y_pred)

    def compile_model(self, lr=0.001):
        """Compiles the model with Dice Loss instead of BCE."""
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=lr),
            # Use the custom Dice Loss to beat the class imbalance
            loss=self.dice_loss,
            # Track the Dice Coefficient so we can see it rise toward 1.0 in TensorBoard
            metrics=[self.dice_coef]
        )

    def train(self, train_gen, val_gen, epochs=100, steps_per_epoch=300, callbacks=None):
        return self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=steps_per_epoch // 5,
            callbacks=callbacks
        )