import os
import matplotlib.pyplot as plt
import numpy as np

# Import the factory we just built
from dataset_generator import AdaCFARDataFactory


def visualize_1d_sample(config_dir, num_targets=2, snr_db=15, clutter_mult=10.0):
    print("Initializing 1D Radar Factory...")
    factory = AdaCFARDataFactory(config_dir)

    print(f"Generating sample: {num_targets} Targets, {snr_db}dB SNR, {clutter_mult}x Clutter...")
    profile, mask = factory.generate_sample(num_targets, snr_base_db=snr_db, clutter_multiplier=clutter_mult)

    # Squeeze the (Nrg, 1) arrays back to 1D for plotting
    profile_1d = profile.squeeze()
    mask_1d = mask.squeeze()
    gates = np.arange(len(profile_1d))

    # Set up the plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot 1: The Cluttered Radar Profile
    color = 'tab:blue'
    ax1.set_xlabel('Range Gate Index')
    ax1.set_ylabel('Normalized Amplitude + Noise', color=color)
    ax1.plot(gates, profile_1d, color=color, linewidth=1.5, alpha=0.8, label="1D Radar Profile")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # Plot 2: The Ground Truth Mask (on the same X axis, different Y axis)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Target Mask (0 or 1)', color=color)
    ax2.plot(gates, mask_1d, color=color, linewidth=2, linestyle='--', label="Target Mask")
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(-0.1, 1.1)

    # Add title and legends
    plt.title(f"1D dCFAR Training Sample | Targets: {num_targets} | Base SNR: {snr_db}dB")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Point this to your simulator configs
    config_path = "configs/"

    # Run a stress test: 3 targets, moderate SNR, massive 10x clutter blocks
    visualize_1d_sample(config_path, num_targets=4, snr_db=20, clutter_mult=10.0)