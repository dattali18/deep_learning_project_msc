import os
import sys

import numpy as np
import tensorflow as tf
from scipy.ndimage import binary_dilation

# Ensure we can import the simulator from the root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from radar_signal_simulator.config.radar_config import RadarConfig
from radar_signal_simulator.config.submode_config import SubmodeLibrary
from radar_signal_simulator.config.operating_submode import OperatingSubmode
from radar_signal_simulator.config.target import Target
from radar_signal_simulator.radar_signal.generator import RadarSimulator


class AdaCFARDataFactory:
    def __init__(self, config_dir):
        print(f"Loading configurations from {config_dir}...")
        self.radar_cfg = RadarConfig.load(os.path.join(config_dir, "radar_config.json"))
        self.submodes = SubmodeLibrary.load(os.path.join(config_dir, "submodes.json"))
        self.sim = RadarSimulator(self.radar_cfg, self.submodes)

        self.sm = self.submodes.submodes["option-1"]
        self.C = 299792458.0

        # In 1D, our input size is exactly the number of range gates
        self.n_gates = self.sm.num_range_gates
        self.max_range = (self.n_gates * self.C) / (2 * self.sm.fs)
        self.max_vel = self.radar_cfg.wavelength / (4 * self.sm.pri)

    def process_1d_profile(self, raw_signal):
        """Applies Beamforming, DPC, and Non-Coherent Integration."""
        # 1. Beamform (Sum across antennas)
        beam = raw_signal.sum(axis=0)  # Shape: (Range_Gates, Pulses)

        # 2. DPC (Matched Filter)
        n_pw = int(self.sm.pulse_width * self.sm.fs)
        h = np.ones(n_pw)

        dpc_signal = np.zeros_like(beam, dtype=np.complex64)
        for p in range(beam.shape[1]):
            dpc_signal[:, p] = np.convolve(beam[:, p], h, mode='same')

        # 3. Envelope Detection & Non-Coherent Integration
        mag_signal = np.abs(dpc_signal)
        profile_1d = np.sum(mag_signal, axis=1)  # Shape: (Range_Gates,)

        # Normalize to [0, 1] for clean thresholding
        max_val = np.max(profile_1d)
        if max_val > 0:
            profile_1d = profile_1d / max_val

        return profile_1d

    def generate_sample(self, num_targets, snr_base_db, clutter_multiplier=2.0):
        """Generates a cluttered 1D profile and a clean target mask."""
        beam = OperatingSubmode.from_values("option-1", 0.0, 0.0)
        targets = []

        for _ in range(num_targets):
            r = np.random.uniform(self.max_range * 0.1, self.max_range * 0.7)
            v = np.random.uniform(-self.max_vel * 0.8, self.max_vel * 0.8)
            t = Target.new([r, 0, 0], [v, 0, 0], rcs=np.random.uniform(0.5, 5.0))
            targets.append(t)

        label_mask = np.zeros(self.n_gates, dtype=np.float32)

        # 1. Generate Clean Mask (Logically ORing individual targets)
        if num_targets > 0:
            clean_master = self.process_1d_profile(self.sim.generate(targets, beam, noise_sigma=0.0))

            for t in targets:
                t_sig = self.process_1d_profile(self.sim.generate([t], beam, noise_sigma=0.0))
                # Threshold at 50% of the main lobe
                raw_mask = (t_sig > 0.75).astype(np.float32)
                # Dilate slightly in 1D so the network has a small "hitbox" to find
                thick_mask = binary_dilation(raw_mask, iterations=4).astype(np.float32)
                label_mask = np.maximum(label_mask, thick_mask)
        else:
            clean_master = np.zeros(self.n_gates, dtype=np.float32)

        # 2. Add Base Thermal Noise
        noise_pwr = 10 ** (-snr_base_db / 10.0)
        noise = np.random.rayleigh(scale=np.sqrt(noise_pwr), size=self.n_gates)
        noisy_profile = clean_master + noise

        # 3. Inject Clutter Edges (The CFAR Breaker)
        # Randomly pick 1 to 3 blocks of range bins to fill with high-amplitude clutter
        num_clutter_blocks = np.random.randint(0, 4)
        for _ in range(num_clutter_blocks):
            clutter_width = np.random.randint(20, 40)  # Width of the clutter block
            start_idx = np.random.randint(0, self.n_gates - clutter_width)

            # Add severe Rayleigh clutter to this specific block
            clutter_noise = np.random.rayleigh(scale=np.sqrt(noise_pwr * clutter_multiplier), size=clutter_width)
            noisy_profile[start_idx: start_idx + clutter_width] += clutter_noise

        # Reshape for CNN input: (Nrg, 1)
        return noisy_profile.reshape(self.n_gates, 1).astype(np.float32), \
            label_mask.reshape(self.n_gates, 1).astype(np.float32)


# --- TFRecord Serialization Helpers ---
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def build_tfrecord_dataset(factory: AdaCFARDataFactory, output_path: str, num_samples: int = 10000):
    """Generates samples and serializes them into a high-speed TFRecord file."""
    print(f"Generating {num_samples} samples into {output_path}...")

    with tf.io.TFRecordWriter(output_path) as writer:
        for i in range(num_samples):
            num_targets = np.random.randint(0, 5)
            snr = np.random.uniform(15, 30)  # Base SNR

            # Generate the numpy arrays
            profile, mask = factory.generate_sample(num_targets, snr)

            # Serialize the arrays to raw bytes
            profile_bytes = profile.tobytes()
            mask_bytes = mask.tobytes()

            # Create the TF Example Protocol Buffer
            feature = {
                'profile': _bytes_feature(profile_bytes),
                'mask': _bytes_feature(mask_bytes)
            }
            example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example_proto.SerializeToString())

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{num_samples} samples...")

    print("Dataset generation complete!")


if __name__ == "__main__":
    config_path = "configs/"
    output_dir = "./tfrecords"
    os.makedirs(output_dir, exist_ok=True)

    factory = AdaCFARDataFactory(config_path)

    # Generate a small test set and a large training set
    build_tfrecord_dataset(factory, os.path.join(output_dir, "val.tfrecord"), num_samples=1000)
    build_tfrecord_dataset(factory, os.path.join(output_dir, "train.tfrecord"), num_samples=10000)
