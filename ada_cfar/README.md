# AdaCFAR-1D: Deep Adaptive CFAR for Heterogeneous Radar Clutter

## 1. Abstract & Motivation
In modern radar systems, traditional Constant False Alarm Rate (CFAR) algorithms, such as Cell-Averaging CFAR (CA-CFAR), excel in homogeneous thermal noise. However, they catastrophically fail in heterogeneous environments—specifically at the edges of massive clutter blocks (e.g., ground returns, forest edges). When the CA-CFAR sliding window encounters a clutter edge, it averages the severe clutter into the clean noise, artificially dragging the detection threshold up and completely blinding the radar to real targets nearby. This phenomenon is known as **Target Masking**.

**AdaCFAR-1D** is an end-to-end Deep Learning replacement for standard CFAR. It processes 1D non-coherently integrated radar profiles using a specialized Dilated Convolutional Neural Network. By looking at massive global context simultaneously, AdaCFAR learns to recognize sudden environmental changes (clutter edges) and suppresses them without masking the adjacent physical targets.

## 2. Dataset Generation & Radar Physics
To ensure the neural network learned true radar physics rather than arbitrary mathematical shapes, the training data was generated using a high-fidelity radar signal simulator.

* **The Pipeline:** Raw simulated antenna arrays undergo Beamforming and Digital Pulse Compression (DPC). The complex signals are then envelope-detected and Non-Coherently Integrated (summed over the PRI) into a 1D magnitude profile.
* **Physical Noise Injection ($kTB F$):** Instead of arbitrary normalization, thermal noise is injected into the complex $I$ and $Q$ channels *before* envelope detection based on absolute physical parameters (System Temperature, Bandwidth, and Noise Figure). This preserves the critical $1/R^4$ power scaling of the radar equation.
* **Logarithmic Compression:** The final physical signals are compressed into a dB scale, allowing the neural network to handle the massive dynamic range of real radar returns.
* **Synthetic Clutter:** Massive, localized Rayleigh-distributed amplitude spikes (up to $5.0\times$ the noise power) are injected randomly into the profiles to simulate the severe clutter edges that break traditional algorithms.

## 3. High-Performance Training Engine
To achieve rapid iteration, the project treats Python as a compiled pipeline. The physically rigorous dataset (10,000+ profiles) is generated offline and serialized into binary **TFRecords**. 

During training, the `tf.data` API streams these binary records directly to the GPU via DMA, bypassing the Python Global Interpreter Lock (GIL). Combined with **Mixed Precision Training** (`float16`) and the inherently small memory footprint of 1D convolutions, the network processes batches of 512 profiles simultaneously, resulting in training times of roughly 5 to 7 seconds per epoch.

## 4. Architectural Evolution

### The Baseline: Traditional CA-CFAR
To prove the model's worth, it was benchmarked against a highly optimized 1D CA-CFAR algorithm (`num_train=12`, `num_guard=6`, $P_{fa}=10^{-4}$). As predicted, the CA-CFAR suffered severe Target Masking. At $25$ dB SNR, it achieved only a $67.1\%$ Probability of Detection ($P_d$), missing hundreds of loud targets because they spawned too close to the synthetic clutter edges.

### Iteration 1: The Dilated CNN (AdaCFAR V1)
The first model utilized escalating dilated convolutions (dilation rates of 1, 4, 16, and 64) to ingest both the local Guard cells and the distant Global Clutter simultaneously. 
* **Result:** It successfully defeated Target Masking, achieving high $P_d$ at the clutter edges. 
* **The Flaw:** Trained using Dice Loss and a flat $0.5$ threshold, V1 hallucinated wildly in low-SNR environments, throwing over 1,200 false alarms. Dice Loss failed to penalize "confident" false positives in the background noise.

### Iteration 2: Focal Loss & Residual Skip Connections (AdaCFAR V2)
To build a tracker-ready system, the architecture and mathematics were overhauled:
1.  **Focal Loss:** The loss function was swapped to Focal Loss to handle the severe class imbalance (thousands of noise gates vs. a few target gates). This mathematically punished the network for confident false alarms.
2.  **Wide Matched-Filter Kernel:** The first layer was expanded to `kernel=7` to ingest the entire spread of the DPC pulse simultaneously.
3.  **Residual Skip Connection:** A ResNet-style skip connection was added, piping the pristine local target shape from the first layer directly into the final fusion layer, preventing the deep clutter-sensing network from "forgetting" the pulse shape.

## 5. Final Results & Evaluation
The model was evaluated not on machine learning metrics, but on strict radar systems engineering standards. A "Hit" was defined as any predicted detection contiguous with the physical DPC pulse width.

### The Evolution of AdaCFAR-1D (Probability of Detection | False Alarms)

We ran a test of 1000 simulated profiles at each SNR level, and the results are as follows:

| SNR | CA-CFAR (Baseline Avg) | V1 (High Pd, High FA) | V2 (Over-penalized) | V3 (Balanced Focal) | V4 (Wide + Skip Conn.) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **30dB** | 78.4% \| 285 FA | 87.3% \| 1006 FA | 62.0% \| 0 FA | 78.5% \| 1 FA | **88.5% \| 0 FA** |
| **25dB** | 68.8% \| 287 FA | 91.1% \| 155 FA | 64.1% \| 0 FA | 74.9% \| 1 FA | **82.4% \| 7 FA** |
| **20dB** | 55.6% \| 291 FA | 87.0% \| 488 FA | 59.7% \| 1 FA | 60.9% \| 7 FA | **71.7% \| 3 FA** |
| **15dB** | 41.3% \| 278 FA | 80.5% \| 1605 FA | 50.5% \| 44 FA | 51.9% \| 50 FA | **58.9% \| 7 FA** |

### Conclusion

AdaCFAR-1D V2 decisively outperforms the traditional engineering baseline. It completely neutralizes the Target Masking flaw in heterogeneous clutter (beating CA-CFAR by nearly $10\%$ $P_d$ at high SNR and $17\%$ $P_d$ at lower SNR). Crucially, the combination of Focal Loss and an optimized threshold suppressed the false alarms to near absolute zero, making this model mathematically safe to deploy into a real-time Kalman tracking pipeline.