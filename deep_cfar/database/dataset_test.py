import os
import matplotlib.pyplot as plt
import numpy as np

# Assuming DeepCFARDataset is in database.rad_yolo_dataset
from database.deep_cfar_dataset import DeepCFARDataset


def visualize_generator_sample(config_dir, num_targets=2, snr_db=15):
    """
    Pulls a single sample from the DeepCFARDataset and plots the 
    noisy network input alongside the clean ground-truth mask.
    """
    print(f"Initializing dataset from configs in: {config_dir}")
    ds = DeepCFARDataset(config_dir)

    print(f"Generating sample with {num_targets} target(s) at {snr_db}dB SNR...")
    img, label_mask = ds.get_sample(num_targets=num_targets, snr_db=snr_db)

    # Remove the channel dimension for plotting (256, 256, 1) -> (256, 256)
    img_2d = img.squeeze()
    mask_2d = label_mask.squeeze()

    # Set up the side-by-side plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: The Input (Noisy RD Map) ---
    im1 = ax1.imshow(img_2d, origin='lower', aspect='auto', cmap='viridis')
    ax1.set_title(f"Network Input: Noisy RD Map ({snr_db}dB SNR)")
    ax1.set_xlabel("Doppler Bin")
    ax1.set_ylabel("Range Bin")
    fig.colorbar(im1, ax=ax1, label="Normalized Magnitude")

    # --- Plot 2: The Label (Binary Target Mask) ---
    # Using 'gray' colormap since it's a binary mask (0 = black, 1 = white)
    im2 = ax2.imshow(mask_2d, origin='lower', aspect='auto', cmap='gray')
    ax2.set_title("Ground Truth Label: Target Mask")
    ax2.set_xlabel("Doppler Bin")
    ax2.set_ylabel("Range Bin")
    fig.colorbar(im2, ax=ax2, label="Target Presence (0 or 1)")

    # Add an overlay to visually verify perfect alignment
    # We find the coordinates of the target pixels and plot red dots on the noisy map
    target_y, target_x = np.where(mask_2d == 1)
    if len(target_x) > 0:
        ax1.scatter(target_x, target_y, c='red', s=1, alpha=0.3, label='Mask Overlay')
        ax1.legend(loc='upper right')
        print(f"Success: Found {len(target_x)} target pixels in the mask.")
    else:
        print("Notice: No target pixels found in the mask (num_targets was 0, or target fell out of bounds).")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Adjust this path if you are running from a different directory
    config_path = "configs/"

    # Test with a clear signal first to ensure alignment
    visualize_generator_sample(config_path, num_targets=4, snr_db=12)

    # You can uncomment this to test the "0 targets" edge case
    # visualize_generator_sample(config_path, num_targets=0, snr_db=15)