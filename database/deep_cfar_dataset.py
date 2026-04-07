import os
import numpy as np
from radar_signal_simulator.config.radar_config import RadarConfig
from radar_signal_simulator.config.submode_config import SubmodeLibrary
from radar_signal_simulator.config.operating_submode import OperatingSubmode
from radar_signal_simulator.config.target import Target
from radar_signal_simulator.radar_signal.generator import RadarSimulator
from scipy.ndimage import binary_dilation


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
        beam = OperatingSubmode.from_values("default", 0.0, 0.0)
        targets = []

        for _ in range(num_targets):
            r = np.random.uniform(100, self.max_range * 0.9)
            v = np.random.uniform(-self.max_vel * 0.8, self.max_vel * 0.8)
            # Wide variance in RCS to simulate strong vs weak targets
            t = Target.new([r, 0, 0], [v, 0, 0], rcs=np.random.uniform(0.1, 5.0))
            targets.append(t)

        label_mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        # 1. The Input Signal: Generate ALL targets together
        clean_signal = self.sim.generate(targets, beam, noise_sigma=0.0)

        # 2. The Perfect Mask: Process each target individually
        for t in targets:
            single_sig = self.sim.generate([t], beam, noise_sigma=0.0)
            single_rd = self.generate_rd_map(single_sig)

            # Threshold at 50% to isolate the main lobe
            raw_mask = single_rd > 0.5

            # DILATE the mask to make the target physically larger for the CNN
            # iterations=2 will expand the mask by 2 pixels in all directions
            thick_mask = binary_dilation(raw_mask, iterations=2).astype(np.float32)

            # Logically OR it with the master mask
            label_mask = np.maximum(label_mask, thick_mask)

        # 3. Add noise to the combined signal for the network input
        noisy_signal = self.add_noise_snr(clean_signal, snr_db)
        noisy_rd_map = self.generate_rd_map(noisy_signal)

        return noisy_rd_map.reshape(self.img_size, self.img_size, 1), \
            label_mask.reshape(self.img_size, self.img_size, 1)

    @staticmethod
    def add_noise_snr(signal, snr_db, noise_floor=1e-12):
        P_sig = np.min(np.abs(signal) ** 2)
        P_sig = max(P_sig, noise_floor)
        P_noise = P_sig / (10 ** (snr_db / 10.0))
        sigma = np.sqrt(P_noise)
        noise = (sigma / np.sqrt(2)) * (
                np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape)
        )
        return signal + noise


def data_generator(ds):
    """
    Python generator for tf.data.Dataset.
    Expects an already initialized DeepCFARDataset instance.
    """
    while True:
        # Randomize targets (0 to 3)
        num_targets = np.random.randint(1, 6)

        # Phase 1 Training: Easy mode (15dB to 25dB)
        snr = np.random.uniform(15, 25)

        img, lbl = ds.get_sample(num_targets=num_targets, snr_db=snr)
        yield img, lbl