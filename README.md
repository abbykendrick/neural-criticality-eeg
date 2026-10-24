# Neural Criticality in EEG: Fractal Dimension and Entropy Across Brain-State Transitions

A Python pipeline that measures how the geometric complexity and information content of an EEG signal change when the brain moves between states, such as from a stressed, high-beta state to a calm, alpha-dominant one.

## 1. Research Objectives

Many analyses treat EEG frequency bands as fixed, independent signals. This project takes a **nonlinear dynamical systems** view instead. It models the nervous system as a complex network that may operate near **criticality**, the boundary between ordered and disordered dynamics.

The pipeline is built to test one working hypothesis:

> **Hypothesis.** When neural activity shifts from a stressed, noisy state to a coherent, rhythmic one (as reported during deep meditation), the signal's fractal dimension should decrease, reflecting a move toward a more ordered attractor.

Related questions this framework could support, but does not yet test:

- Whether coherent external inputs, such as sound at specific frequencies or focused attention, can drive that shift.
- How strongly each metric tracks self-reported or behaviorally measured states.

The current version demonstrates the method on synthetic data. Testing the hypothesis requires real recordings (see [Limitations](#6-limitations-and-next-steps)).

## 2. Mathematical Foundations

### A. Higuchi's Fractal Dimension (HFD)

HFD measures the geometric complexity of a time series without discarding scale-invariant information ([Higuchi, 1988](#references)). For an EEG series $X(1), X(2), \dots, X(N)$, the algorithm builds sub-series at interval $k$:

$$X_m^k = \{X(m), X(m+k), X(m+2k), \dots, X(m + \lfloor \tfrac{N-m}{k} \rfloor \cdot k)\}$$

It computes the normalized curve length $L_m(k)$ for each sub-series. The mean curve length follows a power law:

$$\langle L(k) \rangle \propto k^{-D}$$

$D$ is the fractal dimension. Values near **2.0** indicate white-noise-like irregularity. Values near **1.0** indicate a smooth, regular signal, such as a clean rhythm.

### B. Shannon Entropy

Shannon entropy measures how unpredictable the distribution of signal amplitudes is ([Shannon, 1948](#references)):

$$H(X) = -\sum_{i=1}^{n} P(x_i) \log_2 P(x_i)$$

$P(x_i)$ is the estimated probability of amplitude bin $x_i$. Higher entropy means amplitudes are spread more evenly across the range. Lower entropy means they concentrate in fewer bins.

Note that this measures the *distribution* of amplitudes, not their order in time. A perfectly regular sine wave spends most of its time near its peaks, which spreads its amplitudes across the range and can give it higher entropy than noise. See [Results](#5-results-on-synthetic-data).

## 3. Pipeline Architecture

| Component | Purpose |
| --- | --- |
| `NeuralCriticalityPipeline` | Main class: loads data, processes channels, and plots results. |
| `load_synthetic_or_real_data` | Loads an `.edf` recording through [MNE-Python](https://mne.tools). With no file, it generates a synthetic signal (see below). |
| `apply_butterworth_filter` | Zero-phase Butterworth band-pass filter (`scipy.signal.filtfilt`), which avoids phase distortion. |
| `calculate_higuchi_fractal_dimension` | Computes HFD (default `k_max = 10`). |
| `calculate_shannon_entropy` | Computes entropy in bits from a 30-bin amplitude histogram. |
| `process_criticality_transition` | Splits a channel into two halves, filters each, and computes both metrics. |
| `_render_phase_space_plots` | Plots a 2D time-delay embedding, $x(t)$ against $x(t+\delta)$, for each state. |

### Synthetic data

When no file is provided, the pipeline builds a 12-second, 4-channel signal at 160 Hz:

- **First half (stress model):** Gaussian noise plus a 25 Hz beta component.
- **Second half (coherence model):** an 8 Hz alpha sine wave plus low-amplitude noise.

The first half is band-passed at 20–30 Hz and the second at 4–12 Hz before the metrics are computed.

## 4. Installation and Usage

Requires Python 3.9 or later.

```bash
pip install numpy scipy matplotlib mne
python neural_criticality_pipeline.py
```

To analyze a real recording, pass an EDF file path:

```python
from neural_criticality_pipeline import NeuralCriticalityPipeline

pipeline = NeuralCriticalityPipeline(file_path="recording.edf")
pipeline.load_synthetic_or_real_data(num_channels=4)
pipeline.process_criticality_transition(channel_idx=0)
```

**Output:** a two-panel phase-space figure, one panel per state, with the HFD and entropy values in each title.

## 5. Results on Synthetic Data

Channel 0, default settings, five runs with random seeds 0–4:

| Metric | Stress model (20–30 Hz) | Coherence model (4–12 Hz) | Direction |
| --- | --- | --- | --- |
| Higuchi fractal dimension | 1.96–1.98 | 1.18 | Decreases, as hypothesized |
| Shannon entropy (bits) | 4.21–4.56 | 4.63–4.75 | Increases |

HFD separates the two states clearly. Amplitude-histogram entropy does not drop in the coherent state, for the reason described in [Section 2B](#b-shannon-entropy). Measures that account for temporal order, such as **sample entropy** or **permutation entropy** ([Bandt & Pompe, 2002](#references)), are better suited to detecting rhythmic regularity and are the planned replacement.

## 6. Limitations and Next Steps

- **Synthetic data shows the method works, not that the hypothesis is true.** The synthetic signal is built to contain a shift from noise to rhythm. Real recordings are needed for evidence.
- **Different filters per state.** Each half is filtered to a different band, which affects the metrics on its own. A fair comparison would apply the same filters to both states.
- **File formats.** Only EDF is supported. FIF support would need `mne.io.read_raw_fif`.
- **Planned:** run on public datasets such as the [PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/), add statistical testing across subjects, and compare against established measures such as detrended fluctuation analysis.

## References

- Higuchi, T. (1988). Approach to an irregular time series on the basis of the fractal theory. *Physica D: Nonlinear Phenomena*, 31(2), 277–283.
- Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27(3), 379–423.
- Beggs, J. M., & Plenz, D. (2003). Neuronal avalanches in neocortical circuits. *Journal of Neuroscience*, 23(35), 11167–11177.
- Bandt, C., & Pompe, B. (2002). Permutation entropy: a natural complexity measure for time series. *Physical Review Letters*, 88(17), 174102.
- Gramfort, A., et al. (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience*, 7, 267.

## Related Projects

- [bioelectric-morphogenetic-fields](https://github.com/abbykendrick/bioelectric-morphogenetic-fields): a bioelectric cellular automaton modeling tissue self-repair.
- [spectral-network-synchrony](https://github.com/abbykendrick/spectral-network-synchrony): Kuramoto synchronization on random graphs.

## Author

Abby Kendrick · [GitHub](https://github.com/abbykendrick) · [LinkedIn](https://www.linkedin.com/in/abbykendrick)
