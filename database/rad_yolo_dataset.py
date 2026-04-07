import os
import numpy as np
from radar_signal_simulator.config.radar_config import RadarConfig
from radar_signal_simulator.config.submode_config import SubmodeLibrary
from radar_signal_simulator.config.operating_submode import OperatingSubmode
from radar_signal_simulator.config.target import Target
from radar_signal_simulator.radar_signal.generator import RadarSimulator


class DeepCFARDataset:
    def __init__(self, config_dir, img_size=256):
        # Load Configs
        self.radar_cfg = RadarConfig.load(os.path.join(config_dir, "radar_config.json"))
        self.submodes = SubmodeLibrary.load(os.path.join(config_dir, "submodes.json"))
        self.sim = RadarSimulator(self.radar_cfg, self.submodes)

        self.sm = self.submodes.submodes["default"]
        self.img_size = img_size
        self.C = 299792458.0

        # Calculate limits
        self.max_range = (self.img_size * self.C) / (2 * self.sm.fs)
        self.max_vel = self.radar_cfg.wavelength / (4 * self.sm.pri)

    def generate_rd_map(self, signal):
        """Applies DPC and Doppler FFT, normalizes to [0, 1]."""
        n_pw = int(self.sm.pulse_width * self.sm.fs)
        h = np.ones(n_pw)

        beam = signal.sum(axis=0)
        sig_comp = np.zeros_like(beam)
        for p in range(beam.shape[1]):
            sig_comp[:, p] = np.convolve(beam[:, p], h, mode='same')

        rd = np.fft.fftshift(np.fft.fft(sig_comp, axis=1), axes=1)
        rd_mag = np.abs(rd)

        # Avoid division by zero on empty noise maps
        max_val = np.max(rd_mag)
        if max_val == 0:
            return rd_mag

        return (rd_mag - np.min(rd_mag)) / (max_val + 1e-12)

    def get_sample(self, num_targets: int = 1, snr_db: float | int = 20):
        # 1. Setup Operating Submode
        beam = OperatingSubmode.from_values("default", 0.0, 0.0)
        targets = []

        # 2. Generate N targets
        for _ in range(num_targets):
            r = np.random.uniform(100, self.max_range * 0.9)
            v = np.random.uniform(-self.max_vel * 0.8, self.max_vel * 0.8)
            t = Target.new([r, 0, 0], [v, 0, 0], rcs=np.random.uniform(0.5, 2.0))
            targets.append(t)

        # 3. Generate Clean Signal & Mask
        if num_targets > 0:
            # Generate pure signal without noise
            clean_signal = self.sim.generate(targets, beam, noise_sigma=0.0)
            clean_rd_map = self.generate_rd_map(clean_signal)

            # The Label: A binary mask of the clean target peaks.
            # Using > 0.5 captures the main lobe of the normalized sinc pulse.
            label_mask = (clean_rd_map > 0.5).astype(np.float32)
        else:
            # If 0 targets, create an empty signal and an empty mask
            shape = (self.radar_cfg.array.rows * self.radar_cfg.array.cols,
                     self.sm.num_range_gates,
                     self.sm.num_pulses)
            clean_signal = np.zeros(shape, dtype=np.complex64)
            label_mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        # 4. Generate Noisy Signal (The Network Input)
        # Add noise to the clean signal
        noisy_signal = add_noise_snr(clean_signal, snr_db)
        noisy_rd_map = self.generate_rd_map(noisy_signal)

        # 5. Return matched dimensions: Input (256, 256, 1) and Mask (256, 256, 1)
        return noisy_rd_map.reshape(self.img_size, self.img_size, 1), \
            label_mask.reshape(self.img_size, self.img_size, 1)


def data_generator(ds):
    """
    Python generator for tf.data.Dataset.
    Expects an already initialized DeepCFARDataset instance.
    """
    while True:
        # Randomize targets (0 to 3)
        num_targets = np.random.randint(0, 4)

        # Phase 1 Training: Easy mode (15dB to 25dB)
        snr = np.random.uniform(15, 25)

        img, lbl = ds.get_sample(num_targets=num_targets, snr_db=snr)
        yield img, lbl