import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import tensorflow.keras.backend as K


class AdaCFAR1D:
    def __init__(self, input_shape=(None, 1)):
        # (None, 1) allows it to accept any number of range gates dynamically
        self.input_shape = input_shape
        self.model = self._build_model()

    def _build_model(self):
        """Builds the 1D Dilated Convolutional CFAR Network."""
        inputs = layers.Input(shape=self.input_shape, name="Input_1D_Profile")

        # 1. Local Block
        x = layers.Conv1D(32, 3, padding='same', dilation_rate=1, activation='relu')(inputs)
        x = layers.BatchNormalization(momentum=0.9)(x)  # ADD MOMENTUM HERE

        # 2. Near Training Context
        x = layers.Conv1D(32, 3, padding='same', dilation_rate=4, activation='relu')(x)
        x = layers.BatchNormalization(momentum=0.9)(x)  # AND HERE

        # 3. Far Training Context
        x = layers.Conv1D(32, 3, padding='same', dilation_rate=16, activation='relu')(x)
        x = layers.BatchNormalization(momentum=0.9)(x)  # AND HERE

        # 4. Global Clutter Context
        x = layers.Conv1D(32, 3, padding='same', dilation_rate=64, activation='relu')(x)
        x = layers.BatchNormalization(momentum=0.9)(x)  # AND HERE

        # 5. Fusion Layer (Math processing: CUT vs Background)
        x = layers.Conv1D(16, 1, padding='same', activation='relu', name="Fusion_1x1")(x)

        # 6. Output Threshold Mask
        # NOTE: If using mixed precision, the final output MUST be float32 for numerical stability in the loss function
        outputs = layers.Conv1D(1, 1, padding='same', activation='sigmoid', dtype='float32', name="Output_Mask")(x)

        return models.Model(inputs, outputs, name="AdaCFAR_1D")

    @staticmethod
    def focal_loss(gamma=2.0, alpha=0.25):
        """
        Focal Loss for dense object detection.
        gamma: Focuses the penalty on hard/confident false alarms.
        alpha: Balances the weight between target and background pixels.
        """

        def focal_loss_fixed(y_true, y_pred):
            # Clip predictions to prevent log(0) exploding gradients
            epsilon = K.epsilon()
            y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)

            # Calculate standard Cross Entropy
            cross_entropy = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)

            # Calculate the Focal modulating factor
            p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
            alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
            modulating_factor = K.pow((1.0 - p_t), gamma)

            # Combine and return the mean loss
            return K.mean(alpha_factor * modulating_factor * cross_entropy, axis=-1)

        return focal_loss_fixed

    @staticmethod
    def dice_coef(y_true, y_pred, smooth=1e-6):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    @staticmethod
    def dice_loss(y_true, y_pred):
        return 1.0 - AdaCFAR1D.dice_coef(y_true, y_pred)

    def compile_model(self, lr=0.001):
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=lr),
            loss=self.focal_loss(),
            metrics=[self.dice_coef]
        )


# --- High-Speed TFRecord Parsing ---
def parse_tfrecord_fn(example_proto):
    """Decodes the binary TFRecord back into float32 tensors."""
    feature_description = {
        'profile': tf.io.FixedLenFeature([], tf.string),
        'mask': tf.io.FixedLenFeature([], tf.string),
    }
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)

    # Decode the raw bytes
    profile = tf.io.decode_raw(parsed_features['profile'], tf.float32)
    mask = tf.io.decode_raw(parsed_features['mask'], tf.float32)

    # Reshape back to (Nrg, 1)
    profile = tf.reshape(profile, [-1, 1])
    mask = tf.reshape(mask, [-1, 1])

    return profile, mask


def get_dataset(filepath, batch_size=256):
    """Builds the high-throughput DMA pipeline."""
    # Read the binary files
    raw_dataset = tf.data.TFRecordDataset(filepath)

    # Parse, batch, and prefetch directly to GPU memory
    dataset = raw_dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset