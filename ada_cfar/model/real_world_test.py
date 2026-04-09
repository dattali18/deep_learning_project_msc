import os
import sys
import numpy as np
import tensorflow as tf
from scipy.ndimage import label, center_of_mass
from scipy.ndimage import label

# Add root to sys.path so we can import the database factory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from database.dataset_generator import AdaCFARDataFactory

import numpy as np


def ca_cfar_1d(signal, num_train=10, num_guard=2, pfa=1e-5):
    """
    Executes a 1D Cell-Averaging CFAR (CA-CFAR) using convolution.

    Parameters:
    -----------
    signal : numpy.ndarray
        The 1D magnitude profile of the radar signal.
    num_train : int
        Number of training cells on ONE side of the CUT.
    num_guard : int
        Number of guard cells on ONE side of the CUT.
    pfa : float
        Desired Probability of False Alarm.

    Returns:
    --------
    detections : numpy.ndarray
        Binary array (1 for target, 0 for noise) of the same size as input.
    threshold_mag : numpy.ndarray
        The dynamic threshold curve in magnitude scale (for plotting).
    """
    # 1. Ensure signal is a flat 1D array
    signal_1d = signal.squeeze()

    # 2. Square-Law Detector (Convert magnitude envelope to power)
    power_signal = signal_1d ** 2

    # 3. Calculate scaling factor (alpha)
    N = 2 * num_train
    alpha = N * (pfa ** (-1.0 / N) - 1)

    # 4. Build the sliding window kernel
    kernel_size = 2 * num_train + 2 * num_guard + 1
    kernel = np.ones(kernel_size)

    # Zero out the center (Cell Under Test + Guard Cells)
    center = kernel_size // 2
    kernel[center - num_guard: center + num_guard + 1] = 0

    # Normalize the kernel so convolution calculates the mean automatically
    kernel = kernel / N

    # 5. Calculate local noise floor via convolution
    noise_floor = np.convolve(power_signal, kernel, mode='same')

    # 6. Apply alpha to get the power threshold
    threshold_power = noise_floor * alpha

    # 7. Generate binary detections
    detections = (power_signal > threshold_power).astype(int)

    # Return detections and the threshold converted back to magnitude for plotting
    return detections, np.sqrt(threshold_power)


def extract_centroids(mask_1d):
    """
    Finds contiguous blocks of 1s in the binary mask and returns their exact center indices.
    """
    # label() groups adjacent 1s into uniquely numbered features
    labeled_array, num_features = label(mask_1d)
    centroids = []

    for i in range(1, num_features + 1):
        # Calculate the center of mass for each distinct target blob
        com = center_of_mass(mask_1d, labeled_array, i)
        centroids.append(int(round(com[0])))

    return centroids

def evaluate_profile(true_mask, pred_mask):
    """
    Evaluates Hits and False Alarms based on overlapping contiguous regions.
    - True Positive: A true target block overlaps with ANY predicted block.
    - False Alarm: A continuous predicted block has ZERO overlap with any true target.
    """
    # Group continuous 1s into labeled blocks (1, 2, 3...)
    labeled_true, num_true = label(true_mask)
    labeled_pred, num_pred = label(pred_mask)

    hits = 0
    false_alarms = 0

    # Keep track of which predicted blocks actually hit a target
    # so we don't count them as false alarms later.
    valid_pred_blocks = set()

    # 1. Calculate True Positives (Hits)
    for i in range(1, num_true + 1):
        # Find all range gate indices belonging to this specific true target
        true_indices = np.where(labeled_true == i)[0]

        # Look at what the network predicted at those exact indices
        overlapping_preds = labeled_pred[true_indices]

        # If the network predicted > 0 anywhere in this target's pulse width, it's a hit!
        if np.any(overlapping_preds > 0):
            hits += 1
            # Record which predicted blocks contributed to this hit
            for p_label in np.unique(overlapping_preds):
                if p_label > 0:
                    valid_pred_blocks.add(p_label)

    # 2. Calculate False Alarms
    for j in range(1, num_pred + 1):
        # If a continuous predicted block wasn't validated by ANY true target,
        # the entire block counts as exactly ONE false alarm event.
        if j not in valid_pred_blocks:
            false_alarms += 1

    return num_true, hits, false_alarms


def run_real_world_sweep(model, factory, snr_levels, samples_per_level=200):
    print(f"\n{'=' * 60}")
    print(f"{'SNR (dB)':<10} | {'Targets Generated':<18} | {'Detection Rate (Pd)':<20} | {'Total False Alarms':<18}")
    print(f"{'-' * 60}")

    for snr in snr_levels:
        total_true = 0
        total_hits = 0
        total_fa = 0

        for _ in range(samples_per_level):
            # Randomize targets (include 0 to heavily test false alarms)
            num_targets = np.random.randint(1, 5)

            # Use high clutter multiplier (e.g., 5.0) to stress the network
            profile, true_mask = factory.generate_sample(
                num_targets=num_targets,
                snr_base_db=snr,
                clutter_multiplier=2.0
            )

            # Inference: shape requires (1, Nrg, 1)
            profile_tensor = np.expand_dims(profile, axis=0)
            pred_prob = model(profile_tensor, training=False)

            # Threshold to binary (CFAR activation)
            pred_binary = (pred_prob.numpy().squeeze() > 0.5).astype(int)
            true_binary = true_mask.squeeze().astype(int)

            actual, hits, fas = evaluate_profile(true_binary, pred_binary)

            total_true += actual
            total_hits += hits
            total_fa += fas

        # Calculate Probability of Detection (Pd) safely
        pd = (total_hits / total_true) * 100 if total_true > 0 else 0.0

        print(
            f"{snr:<10} | {total_hits}/{total_true} ({total_true - total_hits} missed) | {pd:>5.1f}%               | {total_fa}")
    print(f"{'=' * 60}\n")

def run_threshold_sweep(model, factory, snr_test=25, samples=300):
    print(f"\n{'=' * 75}")
    print(f" THRESHOLD SWEEP EVALUATION (Fixed SNR: {snr_test}dB | Profiles: {samples})")
    print(f"{'-' * 75}")
    print(f"{'Threshold':<12} | {'Targets Hit':<15} | {'Pd (%)':<10} | {'Total False Alarms':<18}")
    print(f"{'-' * 75}")

    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]

    # Store aggregate results for each threshold
    results = {thresh: {'hits': 0, 'fa': 0, 'true': 0} for thresh in thresholds}

    for i in range(samples):
        num_targets = np.random.randint(0, 5)
        profile, true_mask = factory.generate_sample(
            num_targets=num_targets,
            snr_base_db=snr_test,
            clutter_multiplier=5.0
        )

        profile_tensor = np.expand_dims(profile, axis=0)
        pred_prob = model(profile_tensor, training=False).numpy().squeeze()
        true_binary = true_mask.squeeze().astype(int)

        # Test every threshold on the exact same predicted probabilities
        for thresh in thresholds:
            pred_binary = (pred_prob > thresh).astype(int)
            actual, hits, fas = evaluate_profile(true_binary, pred_binary)

            results[thresh]['true'] += actual
            results[thresh]['hits'] += hits
            results[thresh]['fa'] += fas

    # Print the results
    for thresh in thresholds:
        res = results[thresh]
        total_true = res['true']
        pd = (res['hits'] / total_true) * 100 if total_true > 0 else 0.0
        print(f" > {thresh:<9} | {res['hits']}/{total_true:<8} | {pd:>5.1f}%    | {res['fa']}")

    print(f"{'=' * 75}\n")


def run_head_to_head(model, factory, samples=300):
    model_thr = 0.7

    print(f"\n{'=' * 95}")
    print(f" CA-CFAR vs AdaCFAR-1D HEAD-TO-HEAD (Profiles: {samples})")
    print(f"{'-' * 95}")
    print(f" SNR  | Targets | CA-CFAR Pd | CA-CFAR FA | AdaCFAR Pd (th={model_thr}) | AdaCFAR FA")
    print(f"{'-' * 95}")

    # FIX: Increased num_guard to 6 to prevent Target Self-Masking from the DPC pulse spread
    # Tuned Pfa to 1e-4 for a balanced baseline

    # Sweeping from Loud (25dB) to Quiet (5dB)
    snr_levels = [30, 25, 20, 15, 10, 5]

    for snr in snr_levels:
        total_true = 0
        cfar_hits, cfar_fa = 0, 0
        ada_hits, ada_fa = 0, 0

        for _ in range(samples):
            # Generate the shared reality (Heavy Clutter Edge testing)
            num_targets = np.random.randint(1, 5)
            profile, true_mask = factory.generate_sample(
                num_targets=num_targets,
                snr_base_db=snr,
                clutter_multiplier=5.0
            )

            true_binary = true_mask.squeeze().astype(int)

            # --------------------------------------------------
            # 1. EVALUATE TRADITIONAL CA-CFAR
            # --------------------------------------------------
            cfar_preds, _ = ca_cfar_1d(profile.squeeze(), num_train=12, num_guard=6, pfa=1e-4)
            actual, c_hits, c_fas = evaluate_profile(true_binary, cfar_preds)

            # --------------------------------------------------
            # 2. EVALUATE DEEP LEARNING MODEL (AdaCFAR)
            # --------------------------------------------------
            profile_tensor = np.expand_dims(profile, axis=0)
            pred_prob = model(profile_tensor, training=False).numpy().squeeze()
            # Still using the blunt 0.5 threshold for now
            ada_preds = (pred_prob > model_thr).astype(int)

            _, a_hits, a_fas = evaluate_profile(true_binary, ada_preds)

            # Aggregate stats
            total_true += actual
            cfar_hits += c_hits
            cfar_fa += c_fas
            ada_hits += a_hits
            ada_fa += a_fas

        # Calculate percentages safely
        c_pd = (cfar_hits / total_true) * 100 if total_true > 0 else 0
        a_pd = (ada_hits / total_true) * 100 if total_true > 0 else 0

        # Print the side-by-side row
        print(f" {snr:<2}dB | {total_true:<7} | {c_pd:>8.1f}% | {cfar_fa:<10} | {a_pd:>14.1f}% | {ada_fa}")

    print(f"{'=' * 95}\n")

def main():
    config_path = "../database/configs/"
    model_path = "adacfar_best_v2.keras"

    print("Loading AdaCFAR-1D Model...")
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except OSError:
        print(f"Error: Could not find '{model_path}'.")
        return

    print("Initializing Radar Physics Factory...")
    factory = AdaCFARDataFactory(config_path)

    # We will sweep from easy (25dB) down to incredibly hard (5dB)
    snr_levels = [snr for snr in range(40, 15, -5)]

    # run_real_world_sweep(model, factory, snr_levels, samples_per_level=500)
    # run_threshold_sweep(model, factory, 5, 100)

    run_head_to_head(model, factory, samples=100)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    main()