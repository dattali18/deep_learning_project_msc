---
title: "AdaCFAR-1D: Deep Adaptive Detection for Heterogeneous Radar Clutter"
subtitle: "Final Project Report — Deep Learning Course"
author: "Daniel Attali Sapir Bashan"
date: "June 2026"
geometry: "margin=2cm"
fontsize: "12pt"
---

# 1. Introduction

Radar detection is the first decision stage in a radar processing chain. A radar transmits electromagnetic energy, receives echoes reflected from objects, processes the returned signal, and decides which range cells contain physical targets rather than thermal noise, clutter, or processing artifacts. This decision is critical because every later stage depends on it: missed detections reduce target observability, while false alarms overload the tracking and data association system.

The classical engineering solution for radar detection is **Constant False Alarm Rate** detection, usually abbreviated as **CFAR**. CFAR algorithms estimate the local background level around each **Cell Under Test** and compare the tested cell to an adaptive threshold. In homogeneous noise or homogeneous clutter, this strategy is highly effective because the neighboring cells are statistically representative of the tested cell.

The central limitation appears when the environment is not homogeneous. Near a clutter edge, the CFAR training window may include high-power clutter on one side and clean noise on the other. The averaged background estimate becomes contaminated by the clutter block, the threshold rises, and a physical target near the clutter boundary can be suppressed. This phenomenon is known as **target masking**.
This project investigates a specific hypothesis:

> Classical CA-CFAR can be interpreted as a fixed convolutional detector whose parameters are manually selected from radar theory. This project investigates whether a trainable 1D convolutional neural network can learn a more effective adaptive detection function for heterogeneous clutter environments where the fixed CFAR assumptions are violated.

The goal is not to claim that deep learning should replace CFAR in general. CFAR remains computationally efficient, analytically interpretable, and provides direct control over the false alarm probability. The goal is narrower and more defensible: to demonstrate a proof of concept that, in a specific high-clutter and high-noise detection regime, a dedicated deep learning model can outperform a classical fixed-parameter detector.

The proposed system, **AdaCFAR-1D**, operates on one-dimensional non-coherently integrated radar range profiles. It formulates radar detection as a dense binary segmentation problem over range gates. For each input profile, the network outputs a probability mask indicating which range bins belong to target returns.

The project contains three main components:

1. A physics-inspired radar signal simulator used to generate synthetic training data.
2. A classical CA-CFAR baseline used as the engineering reference.
3. A trainable dilated Conv1D detector designed to learn local target shape and long-range clutter context simultaneously.

# 2. Radar Detection Background

## 2.1 Radar Range Equation

The received power from a monostatic radar target is governed by the radar range equation:

$$
P_r = \frac{P_t G_t G_r \lambda^2 \sigma}{(4\pi)^3 R^4}
$$

where $P_t$ is the transmitted power, $G_t$ and $G_r$ are the transmit and receive antenna gains, $\lambda$ is the wavelength, $\sigma$ is the radar cross section, and $R$ is the target range.

The $R^4$ term is the dominant physical scaling. The transmitted wave spreads over a sphere on the way to the target, and the reflected wave spreads again on the way back. As a result, received target power decreases rapidly with range. A detector trained on radar-like data should therefore not see target amplitudes as arbitrary mathematical peaks. It should see signals generated from a physically meaningful propagation model.

The signal-to-noise ratio is defined as:

$$
\mathrm{SNR} =\frac{P_r}{P_n}
$$

where $P_n$ is the noise power. In a thermal-noise-limited receiver, the noise power is often modeled as:

$$
P_n = k T B F
$$

where $k$ is Boltzmann's constant, $T$ is the system temperature, $B$ is the receiver bandwidth, and $F$ is the receiver noise factor.

## 2.2 Pulsed Radar and Range Bins

A pulsed radar transmits a pulse of duration $\tau$ and samples the received echo over fast time. A received echo from range $R$ appears after the two-way propagation delay:

$$
\tau_R = \frac{2R}{c}
$$

where $c$ is the speed of light. With sampling frequency $f_s$, the sampled range profile is divided into discrete range gates. The range represented by one sample interval is:

$$
\Delta R_{sample} = \frac{c}{2f_s}
$$

For this project, the main simulated submode uses:

| Parameter                |              Value |
| ------------------------ | -----------------: |
| Carrier frequency $f_c$  | $9.6,\mathrm{GHz}$ |
| Sampling frequency $f_s$ |  $10,\mathrm{MHz}$ |
| Pulse width $\tau$       |  $1,\mu\mathrm{s}$ |
| Number of range gates    |             $1024$ |
| Number of pulses         |              $256$ |
| PRF                      | $2000,\mathrm{Hz}$ |

The wavelength is:

$$
\lambda = \frac{c}{f_c}
$$

Substituting $f_c = 9.6,\mathrm{GHz}$ gives:

$$
\lambda  \approx 3.12 \times 10^{-2},\mathrm{m}
$$

The sample spacing in range is:

$$
\Delta R_{sample} = \frac{299792458}{2\cdot 10^7} \approx 14.99,\mathrm{m}
$$

The maximum represented sampled range is:

$$
R_{max,profile} = 1024 \cdot \Delta R_{sample} \approx 15.35,\mathrm{km}
$$

The number of samples covered by the transmitted rectangular pulse is:

$$
N_{\tau} =
\tau f_s =
10^{-6}\cdot 10^7
=
10
$$

This value is important because the matched-filtered target response is not a single-bin object. The detector must identify a spatially extended pulse response over several neighboring range gates.

## 2.3 Doppler Phase Across Pulses

A target with radial velocity $v_r$ induces a Doppler frequency:

$$
f_D
=
\frac{2v_r}{\lambda}
$$

For pulse index $m$ and pulse repetition interval $T_{PRI}$, the Doppler phase is:

$$
\phi_m
=
2\pi \cdot f_D \cdot m \cdot T_{PRI}
$$

The simulated target return across pulses is therefore multiplied by:

$$
e^{j \cdot 2\pi \cdot f_D  \cdot m \cdot T_{PRI}}
$$

Although AdaCFAR-1D ultimately receives a one-dimensional non-coherently integrated range profile, the data generator first creates a complex range-pulse data cube. This preserves the relationship between the physical target velocity and the coherent signal before envelope detection and integration.

## 2.4 Antenna Array Model

The simulator uses a $4 \times 4$ planar antenna array. For a target at azimuth $\theta$ and elevation $\varphi$, the normalized direction components are:

$$
k_x = \cos(\varphi)\cos(\theta)
$$

$$
k_y = \cos(\varphi)\sin(\theta)
$$

For an element located at $(x_n,y_n)$, the array phase is:

$$
\psi_n = \frac{2\pi}{\lambda}
\left(
x_n k_x + y_n k_y
\right)
$$

The array steering vector is:

$$
\mathbf{a}(\theta,\varphi)
\begin{bmatrix}
e^{j\psi_1} &
e^{j\psi_2} &
\cdots &
e^{j\psi_M}
\end{bmatrix}^{T}
$$

where $M=16$ is the number of antenna elements.

The simulated complex radar signal before processing has shape:

$$
\mathbf{S}
\in
\mathbb{C}^{M \times N_r \times N_p}
$$

where $M$ is the number of antenna elements, $N_r$ is the number of range gates, and $N_p$ is the number of pulses.

# 3. Classical CA-CFAR as a Fixed Convolution

## 3.1 CA-CFAR Detection Rule

Cell-Averaging CFAR estimates the local background power around a **Cell Under Test** using training cells while excluding guard cells around the tested bin. Let $x_i$ denote the magnitude profile and let:

$$
p_i = x_i^2
$$

be the square-law detected power. For a CUT at index $i$, CA-CFAR estimates the local noise floor as:

$$
\hat{P}_{n,i}
=
\frac{1}{N}
\sum_{k \in \mathcal{T}_i}
p_k
$$

where $\mathcal{T}_i$ is the set of training cells and $N = |\mathcal{T}_i|$.

The threshold is:

$$
T_i
=
\alpha \hat{P}_{n,i}
$$

The detection decision is:

$$
\hat{y}_i
=
\begin{cases}
1, & p_i > T_i \\
0, & p_i \leq T_i
\end{cases}
$$

For CA-CFAR under the standard exponential-noise power model, the scaling factor is:

$$
\alpha
=
N
\left(
P_{FA}^{-1/N} - 1
\right)
$$

In the project baseline, the CA-CFAR parameters used for evaluation were:

| Parameter                        |     Value |
| -------------------------------- | --------: |
| Training cells per side          |      $12$ |
| Guard cells per side             |       $6$ |
| Total training cells $N$         |      $24$ |
| False alarm probability $P_{FA}$ | $10^{-4}$ |

## 3.2 CFAR as a Convolution Kernel

The CA-CFAR background estimate can be written as a one-dimensional convolution. For example, if a detector uses four training cells on each side and two guard cells on each side, the averaging kernel has the structure:

$$
\mathbf{w}_{CFAR} = \frac{1}{N} [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1]
$$

The zeros correspond to the guard cells and the CUT. The ones correspond to manually selected training cells. The CFAR operation is therefore:

$$
\hat{P}_{n,i}
=
(\mathbf{w}_{CFAR} * \mathbf{p})_i
$$

followed by the threshold comparison:

$$
p_i > \alpha(\mathbf{w}_{CFAR} * \mathbf{p})_i
$$

This interpretation provides the conceptual bridge to deep learning. CA-CFAR is already a convolutional detector, but its kernel is fixed by manual design. The number of training cells, number of guard cells, and threshold factor are chosen before deployment. These choices encode assumptions about the local statistics of the environment.

## 3.3 Failure Mode: Target Masking at Clutter Edges

CA-CFAR works when the training cells are statistically representative of the CUT. Formally, it assumes that the training-cell distribution is close to the local background distribution around the CUT:

$$
P(p_k \mid k \in \mathcal{T}_i)
\approx
P(p_i \mid \text{background})
$$

At a clutter edge, this assumption breaks. One side of the window may contain clean noise while the other side contains high-amplitude clutter. The estimate $\hat{P}_{n,i}$ is then biased upward:

$$
\hat{P}_{n,i}
=
\frac{1}{N}
\left(
\sum_{k \in \mathcal{T}_{clean}} p_k
+
\sum_{k \in \mathcal{T}_{clutter}} p_k
\right)
$$

If the clutter contribution dominates,

$$
\sum_{k \in \mathcal{T}_{clutter}} p_k
\gg
\sum_{k \in \mathcal{T}_{clean}} p_k
$$

then the threshold becomes too high for nearby targets:

$$
T_i = \alpha \hat{P}_{n,i}
$$

A true target can therefore satisfy:

$$
p_i < T_i
$$

even when its physical SNR would be sufficient in a homogeneous environment. This is target masking.
The problem is not that convolution is inappropriate. The problem is that the convolution kernel is fixed and cannot adapt to the clutter geometry.

# 4. Problem Formulation

## 4.1 Detection as Dense 1D Segmentation

AdaCFAR-1D formulates radar detection as a dense segmentation problem over range gates. Each input profile is:

$$
\mathbf{x}
=
\begin{bmatrix}
x_1 & x_2 & \cdots & x_{1024}
\end{bmatrix}^{T}
\in
\mathbb{R}^{1024 \times 1}
$$

The target mask is:

$$
\mathbf{y}
=
\begin{bmatrix}
y_1 & y_2 & \cdots & y_{1024}
\end{bmatrix}^{T}
\in
\{0,1\}^{1024 \times 1}
$$

where $y_i=1$ means range gate $i$ belongs to a physical target response and $y_i=0$ means background.
The neural network learns a function:

$$
f_\theta:
\mathbb{R}^{1024 \times 1}
\rightarrow
[0,1]^{1024 \times 1}
$$

where:

$$
\hat{y}_i
=
f_\theta(\mathbf{x})_i
\cdot
P(y_i=1 \mid \mathbf{x})
$$

The final binary decision is made by thresholding:

$$
\tilde{y}_i
=
\mathbb{1}
\left[
\hat{y}_i > \eta
\right]
$$

where $\eta$ is the inference threshold. The final evaluated model used:

$$
\eta = 0.9
$$

## 4.2 Why This Is Not Ordinary Classification

A profile-level classifier would answer whether a target exists somewhere in the profile. That is insufficient for radar detection because the tracker needs the target location. A regression model could predict one or more target centers, but that would require a separate mechanism for variable target count and clutter rejection.

The dense segmentation formulation avoids these issues. It predicts a detection probability for every range gate, supports zero to multiple targets, and preserves spatial localization. It also makes the model structurally similar to CFAR, because both methods make a local decision for each range gate using surrounding context.

# 5. Dataset Generation and Radar Simulation

## 5.1 Design Philosophy

The dataset was generated synthetically because no real radar data was used. This constraint is important: the project is a proof of concept using a physically motivated simulator, not a validated operational radar detector.

The simulator was designed to avoid training on arbitrary mathematical spikes. Each profile is generated through a radar-inspired signal chain:

1. Complex target signal generation using radar equation amplitude scaling.
2. Antenna array phase response.
3. Beam filtering by field of view.
4. Digital beamforming by summing array channels.
5. Digital pulse compression using a rectangular matched filter.
6. Envelope detection.
7. Non-coherent integration across pulses.
8. Synthetic thermal noise and heterogeneous clutter injection.
9. Binary target-mask generation from the processed clean target response.

This matters because the model learns target responses after radar processing rather than idealized target centers.

## 5.2 Target Signal Generation

Each target is represented by range, azimuth, elevation, radial velocity, and radar cross section. The target is included only if it lies inside the simulated beam field of view:

$$
|\theta_{az,target} - \theta_{az,beam}| \leq \frac{\Theta_{az}}{2}
$$

$$
|\theta_{el,target} - \theta_{el,beam}| \leq \frac{\Theta_{el}}{2}
$$

For a target that passes the field-of-view test, the two-way delay is:

$$
\tau_R = \frac{2R}{c}
$$

The rectangular pulse envelope is:

$$
s_R(t)
=
\begin{cases}
1, & \tau_R \leq t \leq \tau_R + \tau \
0, & \text{otherwise}
\end{cases}
$$

The Doppler modulation across pulses is:

$$
s_D[m]
=
e^{j2\pi f_D mT_{PRI}}
$$

The target range-pulse signal is:

$$
s_{target}[n,m]
=
s_R[n]s_D[m]
$$

The received amplitude is computed from the radar equation:

$$
A_R
=
\sqrt{
\frac{
P_tG_tG_r\lambda^2\sigma
}{
(4\pi)^3R^4
}
}
$$

The full target contribution to the array data cube is:

$$
\mathbf{S}_{target}[:,n,m]
=
\mathbf{a}(\theta,\varphi)
A_R
s_R[n]
s_D[m]
$$

Multiple targets are added linearly:

$$
\mathbf{S}
=
\sum_{q=1}^{Q}
\mathbf{S}_{target}^{(q)}
+
\mathbf{N}
$$

where $Q$ is the number of targets and $\mathbf{N}$ is complex receiver noise when enabled.

## 5.3 Radar Configuration

The primary radar configuration used for the dataset is:

| Parameter             |                  Value |
| --------------------- | ---------------------: |
| Carrier frequency     |     $9.6,\mathrm{GHz}$ |
| Transmit power        |       $10,\mathrm{dB}$ |
| Transmit gain         |       $30,\mathrm{dB}$ |
| Receive gain          |       $30,\mathrm{dB}$ |
| Array size            |           $4 \times 4$ |
| Element spacing $d_x$ |    $0.0156,\mathrm{m}$ |
| Element spacing $d_y$ |    $0.0156,\mathrm{m}$ |
| Azimuth coverage      | $[-60^\circ,60^\circ]$ |
| Elevation coverage    | $[-10^\circ,10^\circ]$ |

The main operating submode is:

| Parameter            |              Value |
| -------------------- | -----------------: |
| Sampling frequency   |  $10,\mathrm{MHz}$ |
| PRF                  | $2000,\mathrm{Hz}$ |
| PRI                  |  $0.5,\mathrm{ms}$ |
| Pulse width          |  $1,\mu\mathrm{s}$ |
| Pulses per profile   |              $256$ |
| Range gates          |             $1024$ |
| Azimuth beam width   |         $20^\circ$ |
| Elevation beam width |          $6^\circ$ |

The maximum unambiguous Doppler velocity implied by the PRF is:

$$
v_{max}
=
\frac{\lambda}{4T_{PRI}}
=
\frac{\lambda f_{PRF}}{4}
$$

Using $\lambda \approx 0.0312,\mathrm{m}$ and $f_{PRF}=2000,\mathrm{Hz}$:

$$
v_{max}
\approx
15.6,\mathrm{m/s}
$$

Targets are sampled with radial velocities inside $80\%$ of this interval:

$$
v_r
\sim
\mathcal{U}(-0.8v_{max},0.8v_{max})
$$

## 5.4 Signal Processing Pipeline

The raw signal generated by the simulator is a complex tensor:

$$
\mathbf{S}
\in
\mathbb{C}^{16 \times 1024 \times 256}
$$

The first processing step is beamforming by summing over antenna elements:

$$
B[n,m] =
\sum_{e=1}^{16}
S[e,n,m]
$$

This produces:

$$
B
\in
\mathbb{C}^{1024 \times 256}
$$

Digital pulse compression is implemented by convolving each pulse with a rectangular matched-filter kernel:

$$
h[n]
\begin{cases}
1, & 0 \leq n < N_{\tau} \\
0, & \text{otherwise}
\end{cases}
$$

where $N_{\tau}=10$ samples for the main submode. For each pulse $m$:

$$
C[:,m]
=
B[:,m] * h
$$

The envelope is then computed:

$$
E[n,m]
=
|C[n,m]|
$$

Finally, non-coherent integration sums the envelope across pulses:

$$
x[n]
=
\sum_{m=1}^{256}
E[n,m]
$$

The resulting vector $\mathbf{x}$ is the one-dimensional range profile used as the network input.

## 5.5 Target Label Generation

The labels are generated from the processed clean target response, not directly from the target center coordinate. This is a key design choice because the target after pulse compression occupies a finite-width response rather than a single range bin.

For each target, a clean target-only profile is generated:

$$
\mathbf{x}^{(q)}_{clean}
=
\mathrm{Process}
\left(
\mathrm{Simulate}
\left(
\text{target } q,\ \mathrm{noise}=0
\right)
\right)
$$

A raw target mask is produced by thresholding the clean response:

$$
m^{(q)}_{raw}[i]
=
\mathbb{1}
\left[
x^{(q)}_{clean}[i] > 0.75
\right]
$$

The mask is then dilated in one dimension:

$$
m^{(q)}_{label}
=
\mathrm{Dilate}
\left(
m^{(q)}_{raw},
4
\right)
$$

The final label is the union of all target masks:

$$
\mathbf{y}
=
\max_q
m^{(q)}_{label}
$$

This makes the target label correspond to the physical pulse-compressed response. It also gives the network a finite detection hitbox, which matches the later evaluation rule: a prediction that overlaps the true target response is counted as a successful hit.

## 5.6 Noise and Clutter Generation

The number of targets per training profile is sampled as:

$$
Q \sim \mathcal{U}\{0,1,2,3,4\}
$$

Each target range is sampled from the inner part of the profile:

$$
R
\sim
\mathcal{U}(0.1R_{max},0.7R_{max})
$$

The radar cross section is sampled as:

$$
\sigma
\sim
\mathcal{U}(0.5,5.0)
$$

The base SNR is sampled during dataset generation as:

$$
\mathrm{SNR}_{base}
\sim
\mathcal{U}(15,30),\mathrm{dB}
$$

The noise power parameter is:

$$
P_n
=
10^{-\mathrm{SNR}_{base}/10}
$$

The background noise added to the one-dimensional magnitude profile is Rayleigh-distributed:

$$
n_i
\sim
\mathrm{Rayleigh}
\left(
\sqrt{P_n}
\right)
$$

Heterogeneous clutter is injected as localized Rayleigh-distributed clutter blocks. The number of clutter blocks is sampled as:

$$
N_c
\sim
\mathcal{U}\{0,1,2,3\}
$$

Each block has random width:

$$
W_c
\sim
\mathcal{U}\{20,\ldots,39\}
$$

Inside a clutter block, additional clutter noise is sampled as:

$$
c_i
\sim
\mathrm{Rayleigh}
\left(
\sqrt{P_n \cdot \kappa}
\right)
$$

where $\kappa$ is the clutter multiplier. The training generator uses $\kappa=2.0$ by default, while the head-to-head stress evaluation uses $\kappa=5.0$.

The final noisy profile is:

$$
\mathbf{x}_{noisy}
=
\mathbf{x}_{clean}
+
\mathbf{n}
+
\mathbf{c}
$$

where $\mathbf{c}$ is nonzero only inside the clutter blocks.

## 5.7 Dataset Size and Serialization

The dataset was generated offline and serialized as TFRecords. Each example contains two raw float32 arrays:

| Field     |      Shape | Meaning                   |
| --------- | ---------: | ------------------------- |
| `profile` | $(1024,1)$ | Noisy radar range profile |
| `mask`    | $(1024,1)$ | Binary target mask        |

The generated split was:

| Split      | Number of profiles |
| ---------- | -----------------: |
| Training   |         $25{,}000$ |
| Validation |          $1{,}000$ |

%% > **⚠️ TODO (Daniel):** Add the final independent test-set generation details if a separate held-out test file was used. The current evaluation script generates test profiles procedurally rather than loading a fixed test TFRecord, so the report should clarify whether the final results are from a deterministic saved test set or from repeated stochastic evaluation.
%%

# 6. Deep Learning Method

## 6.1 Architectural Principle

The architecture is based on the observation that CA-CFAR is a fixed convolutional detector. AdaCFAR-1D keeps the convolutional structure but replaces manually chosen averaging kernels with learned Conv1D filters.

The network is not a generic image model applied to radar. Its structure mirrors the CFAR detection problem:

| Classical CFAR concept       | AdaCFAR-1D analogue          |
| ---------------------------- | ---------------------------- |
| CUT                          | Current range gate           |
| Guard cells                  | Local convolutional context  |
| Training cells               | Dilated neighborhood context |
| Extended clutter environment | Large-dilation convolution   |
| Noise estimate               | Learned feature maps         |
| Threshold comparison         | Sigmoid output probability   |

The model learns both the local morphology of pulse-compressed targets and the broader clutter environment around them.

## 6.2 Dilated Convolutions

A standard 1D convolution with kernel size $K$ computes:

$$
z_i
=
\sum_{r=0}^{K-1}
w_r x_{i+r}
$$

A dilated convolution with dilation rate $d$ computes:

$$
z_i
=
\sum_{r=0}^{K-1}
w_r x_{i+dr}
$$

The receptive field grows without a proportional increase in parameters. For a stack of stride-one convolution layers with kernel sizes $K_l$ and dilation rates $d_l$, the receptive field is:

$$
R
=
1
+
\sum_{l=1}^{L}
(K_l-1)d_l
$$

The first AdaCFAR version used dilation rates:

$$
1,\ 4,\ 16,\ 64
$$

With kernel size $3$ in each layer, this produces an approximate receptive field:

$$
R
=
1
+
2(1+4+16+64)
=
171
$$

This gives each output bin access to a wide neighborhood around the CUT while keeping the parameter count small.

## 6.3 AdaCFAR V1: Dilated CNN

The first architecture used four dilated Conv1D blocks:

| Layer                  | Filters | Kernel | Dilation | Activation |
| ---------------------- | ------: | -----: | -------: | ---------- |
| Local block            |    $32$ |    $3$ |      $1$ | ReLU       |
| Near context           |    $32$ |    $3$ |      $4$ | ReLU       |
| Far context            |    $32$ |    $3$ |     $16$ | ReLU       |
| Global clutter context |    $32$ |    $3$ |     $64$ | ReLU       |
| Fusion                 |    $16$ |    $1$ |      $1$ | ReLU       |
| Output                 |     $1$ |    $1$ |      $1$ | Sigmoid    |

Batch normalization was applied after each main convolutional block.

The V1 model demonstrated that the dilated architecture could recover targets near clutter edges better than CA-CFAR, but it produced many false alarms. The initial loss emphasized overlap with the target mask and did not sufficiently penalize confident detections in the overwhelming background class.

## 6.4 Dice Loss

The Dice coefficient is:

$$
D(\mathbf{y},\hat{\mathbf{y}})
=
\frac{
2\sum_i y_i\hat{y}_i + \epsilon
}{
\sum_i y_i + \sum_i \hat{y}_i + \epsilon
}
$$

The Dice loss is:

$$
\mathcal{L}_{Dice}
=
1
=
D(\mathbf{y},\hat{\mathbf{y}})
$$

Dice loss is useful for imbalanced segmentation problems because it directly optimizes mask overlap. However, in this radar task it had a critical weakness: the background contains many more range gates than the target class, and even a small number of confident false positive regions can be operationally unacceptable.

A model that detects the correct targets but also creates many false alarms is not tracker-ready. In radar systems, false alarms are not merely a classification error; they generate plots that must be gated, associated, and either rejected or turned into tentative tracks.

## 6.5 Focal Loss

The final models used focal loss. For binary classification, define:

$$
p_t
=
y\hat{y}
+
(1-y)(1-\hat{y})
$$

The focal loss is:

$$
\mathcal{L}_{Focal}
=
-\alpha_t
(1-p_t)^\gamma
\log(p_t)
$$

where:

$$
\alpha_t
=
\alpha y + (1-\alpha)(1-y)
$$

The implementation used:

$$
\gamma = 2.0
$$

$$
\alpha = 0.25
$$

Focal loss down-weights easy examples and focuses training on hard errors. In this task, the most important hard errors are confident false detections in clutter and missed target responses near clutter boundaries.

## 6.6 AdaCFAR Final Architecture

The final architecture combines a wide local target extractor with a dilated clutter-sensing path and a residual skip connection.

| Layer                  | Filters | Kernel | Dilation | Output channels |
| ---------------------- | ------: | -----: | -------: | --------------: |
| Input                  |       — |      — |        — |             $1$ |
| Pulse extractor        |    $64$ |    $7$ |      $1$ |            $64$ |
| Near clutter context   |    $64$ |    $3$ |      $4$ |            $64$ |
| Mid clutter context    |    $64$ |    $3$ |     $16$ |            $64$ |
| Global clutter context |    $64$ |    $3$ |     $64$ |            $64$ |
| Residual add           |       — |      — |        — |            $64$ |
| Fusion                 |    $32$ |    $1$ |      $1$ |            $32$ |
| Output                 |     $1$ |    $1$ |      $1$ |             $1$ |

The first layer uses kernel size $7$ rather than $3$. This is a radar-motivated design decision: after pulse compression and mask dilation, the target response occupies multiple neighboring range bins. A wider first kernel gives the model direct access to local pulse morphology.

The dilated path builds a clutter-context representation. The residual skip connection adds the local pulse features back to the deep clutter features:

$$
\mathbf{F}_{fused}
=
\mathbf{F}_{local}
+
\mathbf{F}_{global}
$$

This has a direct radar interpretation. The local path preserves the shape of the target response, while the global path estimates the surrounding clutter environment. The final layers combine both before making a per-bin detection decision.

## 6.7 Parameter Count

The final model is intentionally small. The parameter count can be derived layer by layer.

| Component            |                        Shape |          Count |
| -------------------- | ---------------------------: | -------------: |
| Conv pulse extractor |  $7 \times 1 \times 64 + 64$ |          $512$ |
| BatchNorm            |      $2 \times 64$ trainable |          $128$ |
| Conv near context    | $3 \times 64 \times 64 + 64$ |     $12{,}352$ |
| BatchNorm            |      $2 \times 64$ trainable |          $128$ |
| Conv mid context     | $3 \times 64 \times 64 + 64$ |     $12{,}352$ |
| BatchNorm            |      $2 \times 64$ trainable |          $128$ |
| Conv global context  | $3 \times 64 \times 64 + 64$ |     $12{,}352$ |
| BatchNorm            |      $2 \times 64$ trainable |          $128$ |
| Fusion Conv1D        | $1 \times 64 \times 32 + 32$ |      $2{,}080$ |
| BatchNorm            |      $2 \times 32$ trainable |           $64$ |
| Output Conv1D        |   $1 \times 32 \times 1 + 1$ |           $33$ |
| **Total trainable**  |                            — | **$40{,}257$** |

The total number of trainable parameters is approximately:

$$
N_{\theta}
\approx
4.0 \times 10^4
$$

This is small compared to common image-based CNN architectures. The small parameter count is appropriate for the available synthetic dataset size and reduces the risk of learning arbitrary profile artifacts instead of the intended detection rule.

# 7. Training Setup

## 7.1 Data Pipeline

The dataset is stored as TFRecords and loaded using TensorFlow's `tf.data` API. Each record is parsed into a profile-mask pair, batched, and prefetched:

$$
(\mathbf{x},\mathbf{y})
\rightarrow
\mathrm{batch}
\rightarrow
\mathrm{prefetch}
$$

This avoids repeatedly generating radar profiles during training and reduces Python overhead. Mixed precision training was enabled using TensorFlow's `mixed_float16` policy.

The output layer was explicitly set to `float32`. This is important because the sigmoid output and focal loss involve logarithms. Keeping the final output in float32 improves numerical stability.

## 7.2 Optimization

The final model was trained with Adam:

$$
\theta_{t+1}
=
\theta_t
\eta
\frac{\hat{\mathbf{m}}_t}
{\sqrt{\hat{\mathbf{v}}_t}+\epsilon}
$$

The initial learning rate was:

$$
\eta_0 = 10^{-3}
$$

The batch size was:

$$
B = 256
$$

The maximum epoch count was:

$$
E_{max} = 300
$$

The number of training steps per epoch was:

$$
\left\lfloor
\frac{25000}{256}
\right\rfloor
=
97
$$

The number of validation steps per epoch was:

$$
\left\lfloor
\frac{1000}{256}
\right\rfloor
=
3
$$

The training process used three callbacks:

| Callback          | Monitor         | Purpose                                     |
| ----------------- | --------------- | ------------------------------------------- |
| ReduceLROnPlateau | Validation loss | Reduce learning rate when validation stalls |
| EarlyStopping     | Validation loss | Stop training after no improvement          |
| ModelCheckpoint   | Validation loss | Save best model                             |

The ReduceLROnPlateau factor was $0.5$ with patience $6$, and the minimum learning rate was $10^{-6}$. Early stopping used patience $10$ and restored the best validation weights.

%% > **⚠️ TODO (Daniel):** Add the hardware used for final training, including GPU model, CPU, RAM if available, and approximate training time per epoch. Earlier project notes mention fast epochs, but the final report should state only the measured final value.
%%

The hardware used in training the model is MacBook Pro M5 with MPS acceleration engine for DL.

## 7.3 Inference Threshold

The network outputs probabilities, not hard detections. A threshold sweep was used to select the final decision threshold:

$$
\tilde{y}_i
=
\mathbb{1}
[
\hat{y}_i > \eta
]
$$

The final reported head-to-head evaluation used:

$$
\eta = 0.9
$$

This high threshold reflects the operational importance of suppressing false alarms. In radar tracking, false detections can create false tracks, increase association ambiguity, and consume downstream computational resources.

# 8. Evaluation Methodology

## 8.1 Radar-Oriented Metrics

Per-bin accuracy is not an appropriate primary metric for this problem. Most range bins are background, so a detector can achieve high accuracy by predicting background everywhere. The relevant radar metrics are probability of detection and false alarm count.

The probability of detection is:

$$
P_D
=
\frac{
N_{hit}
}{
N_{target}
}
$$

where $N_{target}$ is the number of true target components and $N_{hit}$ is the number of true target components overlapped by at least one predicted detection component.

False alarms are counted as connected predicted components that do not overlap any true target component. This is more meaningful than counting false-positive bins because a contiguous false detection region would normally become one radar plot or one false alarm event.

## 8.2 Connected-Component Hit Rule

Let the true binary mask be decomposed into connected components:

$$
\mathcal{Y}
=
{Y_1,Y_2,\ldots,Y_M}
$$

Let the predicted binary mask be decomposed into connected components:

$$
\hat{\mathcal{Y}}
=
{\hat{Y}_1,\hat{Y}_2,\ldots,\hat{Y}_K}
$$

A true target component $Y_m$ is counted as detected if there exists a predicted component $\hat{Y}_k$ such that:

$$
Y_m \cap \hat{Y}_k \neq \emptyset
$$

Therefore:

$$
N_{hit}
=
\sum_{m=1}^{M}
\mathbb{1}
\left[
\exists k:
Y_m \cap \hat{Y}_k \neq \emptyset
\right]
$$

A predicted component is counted as a false alarm if it overlaps no true target component:

$$
N_{FA}
=
\sum_{k=1}^{K}
\mathbb{1}
\left[
\forall m:
\hat{Y}_k \cap Y_m = \emptyset
\right]
$$

This rule matches the physics of the generated labels. Since the target mask spans the pulse-compressed response, the detector does not need to identify the exact center bin. It must produce a detection overlapping the physical target response.

## 8.3 Head-to-Head Evaluation Protocol

The final comparison used stochastic test profiles generated from the same physics-based factory but with stronger clutter stress than the default training generator. The evaluation used:

| Parameter                       |                     Value |
| ------------------------------- | ------------------------: |
| Profiles per SNR                |                    $1000$ |
| SNR levels                      | $30,25,20,15,\mathrm{dB}$ |
| Targets per profile             |                $1$ to $4$ |
| Clutter multiplier              |                     $5.0$ |
| AdaCFAR threshold               |                     $0.9$ |
| CA-CFAR training cells per side |                      $12$ |
| CA-CFAR guard cells per side    |                       $6$ |
| CA-CFAR $P_{FA}$                |                 $10^{-4}$ |

The use of $1$ to $4$ targets per evaluation profile removes empty-profile cases from the head-to-head detection-rate comparison and focuses the test on detection performance under clutter.

# 9. Results

## 9.1 Architecture Evolution

The project evaluated multiple model variants and training strategies. The progression is summarized below.

| Model            | Main change                         | Observed behavior                                                 |
| ---------------- | ----------------------------------- | ----------------------------------------------------------------- |
| CA-CFAR baseline | Fixed analytical kernel             | Strong false alarm control but target masking near clutter edges  |
| AdaCFAR V1       | Dilated Conv1D with Dice loss       | High detection probability but excessive false alarms             |
| AdaCFAR V2       | Focal loss                          | Strong false alarm suppression but overly conservative detections |
| AdaCFAR V3       | Balanced focal/threshold tuning     | Improved trade-off between detection and false alarms             |
| AdaCFAR V4       | Wide first kernel and residual skip | Best final trade-off                                              |

The key lesson is that solving target masking alone is not sufficient. A detector that recovers targets near clutter edges but creates many false alarms is not acceptable for a radar processing chain. The final architecture therefore optimizes the balance between detection probability and false alarm suppression.

## 9.2 Final Head-to-Head Results

The final evaluation used $1000$ simulated profiles at each SNR level. The table reports probability of detection and total false alarm events.

|              SNR |     CA-CFAR Baseline | V1 High $P_D$, High FA |   V2 Over-penalized |   V3 Balanced Focal |         V4 Wide + Skip |
| ---------------: | -------------------: | ---------------------: | ------------------: | ------------------: | ---------------------: |
| $30,\mathrm{dB}$ | $78.4\%$ \| $285$ FA |  $87.3\%$ \| $1006$ FA |  $62.0\%$ \| $0$ FA |  $78.5\%$ \| $1$ FA | **$88.5\%$ \| $0$ FA** |
| $25,\mathrm{dB}$ | $68.8\%$ \| $287$ FA |   $91.1\%$ \| $155$ FA |  $64.1\%$ \| $0$ FA |  $74.9\%$ \| $1$ FA | **$82.4\%$ \| $7$ FA** |
| $20,\mathrm{dB}$ | $55.6\%$ \| $291$ FA |   $87.0\%$ \| $488$ FA |  $59.7\%$ \| $1$ FA |  $60.9\%$ \| $7$ FA | **$71.7\%$ \| $3$ FA** |
| $15,\mathrm{dB}$ | $41.3\%$ \| $278$ FA |  $80.5\%$ \| $1605$ FA | $50.5\%$ \| $44$ FA | $51.9\%$ \| $50$ FA | **$58.9\%$ \| $7$ FA** |

The final model improves detection probability over CA-CFAR at every evaluated SNR while strongly suppressing false alarms.

The absolute improvement in $P_D$ for the final model is:

|              SNR | CA-CFAR $P_D$ | V4 $P_D$ | Absolute improvement |
| ---------------: | ------------: | -------: | -------------------: |
| $30,\mathrm{dB}$ |      $78.4\%$ | $88.5\%$ |       $+10.1$ points |
| $25,\mathrm{dB}$ |      $68.8\%$ | $82.4\%$ |       $+13.6$ points |
| $20,\mathrm{dB}$ |      $55.6\%$ | $71.7\%$ |       $+16.1$ points |
| $15,\mathrm{dB}$ |      $41.3\%$ | $58.9\%$ |       $+17.6$ points |

The improvement grows as SNR decreases. This suggests that the learned detector is not merely exploiting high-amplitude target peaks. It is using contextual structure to recover targets that are difficult for the fixed CFAR window.

## 9.3 Interpretation of V1

V1 achieved high probability of detection but generated many false alarms. This is consistent with the loss function and model objective. Dice loss rewards overlap with true target regions, but in a profile with $1024$ range gates and only a few target regions, the operational cost of false-positive connected components must be penalized directly.

At $15,\mathrm{dB}$, V1 achieved $80.5\%$ detection probability, much higher than CA-CFAR, but produced $1605$ false alarm events. Such a detector would be unsuitable for a tracking pipeline because it would generate many false plots.

## 9.4 Interpretation of V2 and V3

The focal-loss variants reduced false alarms dramatically. V2 became overly conservative, reaching zero false alarms at high SNR but losing many true detections. This shows that false alarm suppression alone is not the desired objective.

V3 improved the balance between target recovery and false alarm suppression. It demonstrated that the decision threshold is an important part of the detector design. In a dense segmentation detector, the sigmoid output should not automatically be thresholded at $0.5$. The threshold must reflect the downstream system cost of false alarms versus missed detections.

## 9.5 Interpretation of V4

The final architecture achieved the best trade-off. The wide first convolution improves sensitivity to the physical shape of the pulse-compressed target response. The residual skip connection preserves that local target information while the dilated path learns the broader clutter context.

At $20,\mathrm{dB}$, CA-CFAR achieved $55.6\%$ detection probability with $291$ false alarms, while V4 achieved $71.7\%$ detection probability with only $3$ false alarms. This result directly supports the central hypothesis: a learned convolutional detector can outperform a fixed CFAR-like convolution in the specific heterogeneous clutter regime tested here.

## 9.6 Figures

<!-- ![Example radar profile and target mask](https://chatgpt.com/c/images/image1.png) -->
<!-- ![Example cluttered profile and detector behavior](https://chatgpt.com/c/images/image2.png) -->

![training-example-1](image1.png)
![training-example-2](image2.png)

%% > **⚠️ TODO (Daniel):** Add final figure captions after checking the exact contents of `images/image1.png` and `images/image2.png`. Captions should state what is plotted, what the axes mean, and which model or baseline is shown.

> **⚠️ TODO (Daniel):** Add at least one CA-CFAR failure figure and one AdaCFAR success figure. The strongest figure would show the noisy profile, CA-CFAR threshold, true target mask, and AdaCFAR probability output on the same range axis.
> **⚠️ TODO (Daniel):** Add training and validation loss curves if the model is retrained. If the original training logs are unavailable, retrain the final model and export the history to CSV so the report can include reproducible training curves.
> %%

# 10. Discussion

## 10.1 What the Project Demonstrates

The results support the claim that learned convolutional detectors can be useful in radar environments where fixed CFAR assumptions break down. The core advantage is not that the model is “deep” in a generic sense. The advantage is that it learns the detection kernel and nonlinear decision rule from examples of the specific environment.

CA-CFAR uses a manually designed window. It assumes that the selected training cells provide a reliable estimate of the local background. AdaCFAR-1D learns multiple contextual filters across different scales. This allows it to respond differently to local pulse-like structures, smooth noise regions, and abrupt clutter discontinuities.

The final model should be interpreted as a proof of concept for specialized learned detection under heterogeneous clutter. It is not a universal radar detector and does not replace the need for classical CFAR theory.

## 10.2 Why False Alarm Counting Matters

A radar detector is not evaluated only by detecting targets. Every false alarm can become a false plot, and every false plot can create downstream association ambiguity. A high false alarm rate can overload a tracker even if the detector has high probability of detection.

This is why V1 is not the best model despite its strong $P_D$. The final architecture is better because it improves detection probability while preserving near-zero false alarm behavior across the tested SNR levels.

## 10.3 Why Synthetic Data Is Both a Strength and a Limitation

Synthetic data is a strength because it enables controlled experiments. The simulator provides exact ground truth, controlled SNR, controlled clutter severity, and repeatable evaluation across target counts and clutter conditions. It also allows the dataset to be generated through a physically meaningful radar processing chain rather than arbitrary peak injection.

Synthetic data is also the main limitation. Real clutter contains spatial structure, temporal correlation, multipath, sidelobes, antenna effects, calibration errors, non-Rayleigh distributions, and environmental dependencies not captured by the current generator. Therefore, the results should not be interpreted as evidence of operational readiness.

The correct conclusion is that the method is promising under the simulated assumptions and deserves further testing under more realistic clutter models.

## 10.4 Computational Considerations

The final model contains approximately $40{,}000$ trainable parameters and uses only one-dimensional convolutions. This makes it computationally lightweight compared to image-based deep learning models. Its input length is fixed at $1024$ bins during training, but the Keras input shape allows variable-length profiles because the convolutional operations are fully convolutional.

The model is therefore structurally compatible with real-time processing constraints, but this was not benchmarked as part of the project. A real deployment study would need to measure inference latency, memory use, batching behavior, and deterministic runtime on the intended hardware.

# 11. Limitations

The main limitations are:

1. The dataset is synthetic and was not validated against real radar recordings.
2. The clutter model uses localized Rayleigh-distributed amplitude blocks rather than measured environmental clutter.
3. The detector operates on one-dimensional range profiles rather than full range-Doppler maps.
4. The evaluation uses procedurally generated test profiles, and the exact random seed should be fixed for full reproducibility.
5. The model does not provide analytical false alarm guarantees like classical CFAR.
6. The final threshold was selected empirically and may require recalibration under distribution shift.
7. The simulator uses a simplified rectangular pulse compression model rather than a full waveform-dependent matched filter.
8. The report currently lacks final training curves unless the model is retrained or logs are recovered.

These limitations do not invalidate the project. They define the boundary of the claim. The project demonstrates that, inside a controlled heterogeneous-clutter simulation, a radar-motivated Conv1D detector can outperform a fixed CA-CFAR baseline.

# 12. Future Work

The immediate next step is to evaluate the method on richer radar representations. A natural extension is a two-dimensional detector operating on range-Doppler maps:

$$
\mathbf{X}
\in
\mathbb{R}^{N_r \times N_d}
$$

This would allow the network to exploit Doppler separation and distinguish stationary clutter from moving targets.

Additional future directions include:

1. Replacing the rectangular pulse model with a waveform-specific matched filter.
2. Adding correlated clutter fields instead of independent Rayleigh clutter blocks.
3. Testing Ordered-Statistic CFAR and Greatest-Of CFAR baselines in addition to CA-CFAR.
4. Calibrating the simulator against realistic radar parameter ranges.
5. Measuring inference latency on GPU and CPU.
6. Studying threshold calibration under distribution shift.
7. Connecting the detector output to a Kalman-filter-based tracking pipeline.
8. Training with domain randomization to improve robustness to unseen clutter statistics.

# 13. AI Usage Disclosure

AI tools were used during the project workflow for brainstorming, code review, explanation of deep learning design options, and assistance in drafting the final report. The simulator, model code, evaluation code, project-specific results, and engineering decisions were provided and reviewed by the author. AI assistance was used as a writing and reasoning aid rather than as a source of experimental results.

The use of AI is disclosed because the course instructions explicitly permit AI assistance when the usage is explained.

# 14. Conclusion

This project studied target detection in high-clutter, high-noise radar environments. The starting point was the observation that CA-CFAR can be viewed as a fixed convolutional detector: it applies a manually designed averaging kernel around each cell under test and compares the result to a threshold. This design is effective in homogeneous environments but degrades near clutter edges, where the training cells no longer represent the local background.

AdaCFAR-1D replaces the fixed CFAR kernel with a trainable one-dimensional convolutional network. The architecture was designed around the radar structure of the problem: local convolutions preserve the pulse-compressed target shape, dilated convolutions capture wider clutter context, and a residual skip connection fuses local target morphology with global background information.

The final model improved probability of detection over CA-CFAR at all tested SNR levels while strongly suppressing false alarm events. At $20,\mathrm{dB}$, the final model improved detection probability from $55.6\%$ to $71.7\%$ while reducing false alarms from $291$ to $3$ in the reported evaluation. At $15,\mathrm{dB}$, it improved detection probability from $41.3\%$ to $58.9\%$ while reducing false alarms from $278$ to $7$.

The conclusion is not that deep learning replaces CFAR as a general radar detector. The correct conclusion is more specific: when the classical fixed-window assumptions are violated by heterogeneous clutter, a dedicated learned convolutional detector can learn a better task-specific detection rule. This makes AdaCFAR-1D a successful proof of concept for deep learning as an adaptive radar detection component in environments where hand-designed fixed-parameter methods suffer known failure modes.

# 15. Code Availability

%% > **⚠️ TODO (Daniel):** Add the final repository link or code submission path. The course requirements ask for code that enables running the full project, including the dataset or a link to the dataset.
%%
Link to the source code (Simulation + Model training + testing) [here](https://github.com/dattali18/deep_learning_project_msc)

# 16. References

- M. I. Skolnik, _Introduction to Radar Systems_, 3rd ed., McGraw-Hill, 2001.
- M. A. Richards, J. A. Scheer, and W. A. Holm, _Principles of Modern Radar: Basic Principles_, SciTech Publishing, 2010.
- N. Levanon and E. Mozeson, _Radar Signals_, Wiley-IEEE Press, 2004.
- H. Rohling, “Radar CFAR Thresholding in Clutter and Multiple Target Situations,” _IEEE Transactions on Aerospace and Electronic Systems_, 1983.
- P. P. Gandhi and S. A. Kassam, “Analysis of CFAR Processors in Nonhomogeneous Background,” _IEEE Transactions on Aerospace and Electronic Systems_, 1988.
- S. Haykin, _Neural Networks and Learning Machines_, Pearson, 2009.
- I. Goodfellow, Y. Bengio, and A. Courville, _Deep Learning_, MIT Press, 2016.
- Y. LeCun, Y. Bengio, and G. Hinton, “Deep Learning,” _Nature_, 2015.
- F. Yu and V. Koltun, “Multi-Scale Context Aggregation by Dilated Convolutions,” _International Conference on Learning Representations_, 2016.
- K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” _IEEE Conference on Computer Vision and Pattern Recognition_, 2016.
- T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, “Focal Loss for Dense Object Detection,” _IEEE International Conference on Computer Vision_, 2017.
- D. P. Kingma and J. Ba, “Adam: A Method for Stochastic Optimization,” _International Conference on Learning Representations_, 2015.
