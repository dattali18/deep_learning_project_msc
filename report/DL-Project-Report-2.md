# Deep Learning for Radar Detection in Heterogeneous Clutter: AdaCFAR-1D

## 1. Abstract / Executive Summary

Radar detection is the critical first stage in a radar processing chain, dictating the success of downstream data association and tracking. Classical Constant False Alarm Rate (CFAR) algorithms perform well in homogeneous noise but suffer from "target masking" near clutter edges, where averaged background estimates suppress valid physical targets. 

This project introduces AdaCFAR-1D, a lightweight, trainable 1D convolutional neural network that frames radar detection as a dense binary segmentation problem over range gates. By learning both local target morphology and broader clutter context, the final architecture improves the probability of detection at $20\text{ dB}$ SNR from $55.6\%$ (classical CA-CFAR) to $71.7\%$, while simultaneously reducing false alarm events from 291 to just 3.


## 2. Introduction & Problem Formulation

Classical Cell-Averaging CFAR (CA-CFAR) estimates the local background power around a Cell Under Test (CUT) using a manually selected window of training cells. It is essentially a fixed 1D convolution. For example, a detector with four training cells and two guard cells on each side applies the following kernel:

$$\mathbf{w}_{CFAR} = \frac{1}{N} \begin{bmatrix} 1 & 1 & 1 & 1 & 0 & 0 & 0 & 0 & 0 & 1 & 1 & 1 & 1 \end{bmatrix}$$

The zeros correspond to the guard cells and the CUT, while the ones compute the background average. A detection occurs if the CUT exceeds the background estimate scaled by a threshold factor.

The central failure mode is target masking. At a clutter edge, one side of the CFAR window contains clean noise while the other contains high-power clutter. The background estimate is biased upward by the clutter block, raising the threshold unnaturally high and suppressing valid physical targets located near the boundary. Because the CFAR convolution kernel is fixed by manual design, it cannot adapt to this asymmetric clutter geometry.

AdaCFAR-1D investigates whether a learned convolutional detector can outperform this fixed mathematical kernel in heterogeneous environments, without sacrificing computational efficiency or creating excessive false alarms.



## 3. Dataset Generation Overview

Because a radar detector must not train on arbitrary mathematical spikes, the dataset was generated using a physics-inspired pulsed radar simulator ($9.6\text{ GHz}$ carrier, $1\text{ }\mu\text{s}$ pulse width, non-coherent integration). 

The neural network operates on 1D non-coherently integrated range profiles. The simulated environment injects:
* Rayleigh-distributed thermal base noise.
* Heterogeneous clutter (randomly sized blocks of high-power Rayleigh noise).
* Multiple physical targets modeled via the Radar Range Equation, capturing accurate pulse-compressed spatial responses.

Crucially, the target labels are not single bins. Because a pulse-compressed target response physically occupies multiple neighboring range gates, the ground-truth masks are dilated representations of the clean signal. This provides the network with a finite detection hit-box, mimicking the physical constraints of real radar hardware.


## 4. Deep Learning Methodology

### 4.1 Dense 1D Segmentation Framework
AdaCFAR-1D frames detection as dense segmentation over the 1024 range gates of a profile. It mirrors the CFAR concept but replaces the fixed analytical kernel with learned Conv1D filters. 

### 4.2 Architecture Evolution & The Final Model
The architecture underwent four major iterations to solve the balance between target recovery and false alarm suppression:

* **V1 (Dilated CNN):** Achieved high target recovery near clutter but generated unacceptable operational false alarms.
* **V2 & V3 (Loss Tuning):** Transitioned from Dice Loss to Focal Loss to aggressively penalize confident false positives in the overwhelming background class.
* **V4 (Final Architecture):** Combines a wide local target extractor with a dilated clutter-sensing path.

The final model utilizes a wide initial convolution (kernel size 7) to gain direct access to local pulse morphology. A parallel path utilizes dilated convolutions (rates up to 64) to build a wide representation of the surrounding clutter environment. A residual skip connection fuses the local shape feature with the global clutter context before the final decision layer.

### 4.3 Hardware Feasibility & Footprint
The final model contains approximately 40,000 trainable parameters. The architecture relies entirely on 1D convolutions, resulting in a highly predictable memory access pattern and a small computational footprint. This structural simplicity is optimized for low-latency parallelization, making the inference pipeline highly suitable for real-time deployment on GPU architectures via CUDA, or integration into resource-constrained embedded edge environments running C/C++ backends.


## 5. Evaluation & Results

### 5.1 Radar-Oriented Evaluation Metrics
Per-bin accuracy is discarded in favor of radar-appropriate metrics: Probability of Detection ($P_D$) and False Alarm ($FA$) count. 

Evaluation utilizes a Connected-Component Hit Rule:
* **Hit:** A true target component is detected if any part of it is overlapped by a predicted detection mask.
* **False Alarm:** A contiguous predicted mask that overlaps no true target is counted as a single false alarm event (as it would generate a single false plot in a tracker).

### 5.2 Head-to-Head Performance
The final evaluation tested 1,000 simulated profiles per SNR level, utilizing heavy heterogeneous clutter stress (clutter multiplier $= 5.0$). AdaCFAR-1D utilized an inference threshold of $0.9$, optimized for strict false alarm suppression.

| SNR | CA-CFAR Baseline ($P_D$ \| FA) | AdaCFAR-1D Final V4 ($P_D$ \| FA) |
| :--- | :--- | :--- |
| $30\text{ dB}$ | $78.4\%$ \| 285 FA | $88.5\%$ \| 0 FA |
| $25\text{ dB}$ | $68.8\%$ \| 287 FA | $82.4\%$ \| 7 FA |
| $20\text{ dB}$ | $55.6\%$ \| 291 FA | $71.7\%$ \| 3 FA |
| $15\text{ dB}$ | $41.3\%$ \| 278 FA | $58.9\%$ \| 7 FA |

The learned detector consistently outperforms the fixed CA-CFAR window, recovering targets in heterogeneous clutter that CFAR routinely masks, while drastically reducing the number of false tracks that would be sent to a downstream data association system.

> [!NOTE] 
> *TODO: Insert Figure 1 here showing a noisy profile where CA-CFAR's threshold spikes over a valid target, alongside the AdaCFAR probability mask successfully isolating it.*



## 6. Discussion, Limitations & Future Work
This project demonstrates that when classical fixed-window assumptions are violated by heterogeneous clutter, a dedicated convolutional network can learn a superior, task-specific detection rule. The use of Focal Loss and high inference thresholds ensures the network acts as a viable engineering component rather than just a theoretical classifier.

**Limitations:** The primary limitation is the reliance on synthetic data. While physically motivated, the simulator utilizes localized Rayleigh blocks rather than measured environmental clutter (which contains spatial correlation, multipath, and non-Rayleigh distributions). 

**Future Work:** *Transitioning from 1D range profiles to 2D range-Doppler maps to allow the network to exploit velocity separation.
* Benchmarking deterministic runtime and inference latency on target embedded hardware.
* Connecting the detector output to a Kalman-filter-based tracking pipeline to measure the direct impact on track continuity and purity.

## 7. Appendices

### Appendix A: Radar Physics & Simulator Details
The simulator utilizes the following core physical models for dataset generation:

**Received Power (Radar Range Equation):**
$$P_r = \frac{P_t G_t G_r \lambda^2 \sigma}{(4\pi)^3 R^4}$$

**Noise Power:**
$$P_n = k T B F$$

**Doppler Phase Across Pulses:**
$$\phi_m = 2\pi \cdot f_D \cdot m \cdot T_{PRI}$$
Where:
$$f_D = \frac{2v_r}{\lambda}$$

**Antenna Array Model (16-element planar):**
The array steering vector for azimuth $\theta$ and elevation $\varphi$ uses phase shifts:
$$\psi_n = \frac{2\pi}{\lambda} (x_n \cos(\varphi)\cos(\theta) + y_n \cos(\varphi)\sin(\theta))$$

### Appendix B: Network Parameter Detail (Final V4)
| Layer Component | Kernel | Dilation | Output Channels | Trainable Parameters |
| :--- | :--- | :--- | :--- | :--- |
| Conv pulse extractor | 7 | 1 | 64 | 512 + 128 (BN) |
| Conv near context | 3 | 4 | 64 | 12,352 + 128 (BN) |
| Conv mid context | 3 | 16 | 64 | 12,352 + 128 (BN) |
| Conv global context | 3 | 64 | 64 | 12,352 + 128 (BN) |
| Fusion Conv1D | 1 | 1 | 32 | 2,080 + 64 (BN) |
| Output Conv1D | 1 | 1 | 1 | 33 |
| **Total** | | | | **~40,257** |

### Appendix C: References
* M. I. Skolnik, Introduction to Radar Systems, 3rd ed., McGraw-Hill, 2001.
* H. Rohling, "Radar CFAR Thresholding in Clutter and Multiple Target Situations," IEEE Transactions on Aerospace and Electronic Systems, 1983.
* T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal Loss for Dense Object Detection," IEEE International Conference on Computer Vision, 2017.
* F. Yu and V. Koltun, "Multi-Scale Context Aggregation by Dilated Convolutions," International Conference on Learning Representations, 2016.