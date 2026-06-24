import numpy as np
import scipy.signal as signal
import scipy.stats as stats
import matplotlib.pyplot as plt
import mne

class NeuralCriticalityPipeline:
    """
    An open-source signal processing pipeline designed to quantify
    thermodynamic criticality and geometric fractal shifts in multi-channel EEG data.
    """
    def __init__(self, file_path: str = None, sampling_rate: float = 160.0):
        self.file_path = file_path
        self.sfreq = sampling_rate
        self.raw_data = None
        self.channels = []
        
    def load_synthetic_or_real_data(self, duration_seconds: int = 10, num_channels: int = 4):
        """
        Loads authentic EEG datasets via MNE-Python. If no path is provided,
        it generates a dual-state synthetic matrix modeling chaos vs coherence.
        """
        if self.file_path:
            raw = mne.io.read_raw_edf(self.file_path, preload=True)
            self.sfreq = raw.info['sfreq']
            self.channels = raw.info['ch_names'][:num_channels]
            self.raw_data = raw.get_data(picks=self.channels)
        else:
            print("[INFO] No file provided. Constructing synthetic multi-channel matrix...")
            n_samples = int(self.sfreq * duration_seconds)
            t = np.linspace(0, duration_seconds, n_samples)
            self.channels = [f'EEG_CH{i}' for i in range(num_channels)]
            
            # Phase 1: High-noise/Stress state (1/f Pink noise + high Beta jitter)
            noise_state = np.random.normal(0, 1.0, (num_channels, n_samples // 2))
            beta_jitter = np.sin(2 * np.pi * 25 * t[:n_samples // 2]) * 0.5
            state_1 = noise_state + beta_jitter
            
            # Phase 2: Highly coherent, phase-locked state (Dominant Alpha/Theta rhythm)
            alpha_wave = np.sin(2 * np.pi * 8 * t[n_samples // 2:]) * 1.5
            low_noise = np.random.normal(0, 0.2, (num_channels, n_samples // 2))
            state_2 = alpha_wave + low_noise
            
            self.raw_data = np.hstack((state_1, state_2))

    def apply_butterworth_filter(self, data: np.ndarray, lowcut: float, highcut: float, order: int = 4) -> np.ndarray:
        """
        Applies a zero-phase forward-backward digital Butterworth filter to isolate 
        specific informational neural bandwidths.
        """
        nyquist = 0.5 * self.sfreq
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(order, [low, high], btype='band')
        return signal.filtfilt(b, a, data, axis=-1)

    def calculate_higuchi_fractal_dimension(self, time_series: np.ndarray, k_max: int = 10) -> float:
        """
        Calculates Higuchi's Fractal Dimension (HFD) to quantify time-series geometry complexity.
        An orderly, phase-locked rhythm drops toward 1.0; white noise scales toward 2.0.
        """
        N = len(time_series)
        L = np.zeros(k_max)
        
        for k in range(1, k_max + 1):
            Lk = 0
            for m in range(k):
                n_max = int(np.floor((N - m) / k))
                if n_max > 1:
                    sub_seq = np.abs(time_series[m + np.arange(1, n_max) * k] - 
                                     time_series[m + np.arange(0, n_max - 1) * k])
                    L_mk = np.sum(sub_seq) * (N - 1) / (n_max * k * k)
                    Lk += L_mk
            L[k - 1] = Lk / k
            
        x_log = np.log(1.0 / np.arange(1, k_max + 1))
        y_log = np.log(L)
        slope, _ = np.polyfit(x_log, y_log, 1)
        return slope

    def calculate_shannon_entropy(self, time_series: np.ndarray, bins: int = 30) -> float:
        """
        Calculates Shannon Entropy to quantify signal unpredictability and informational chaos.
        """
        hist, _ = np.histogram(time_series, bins=bins, density=True)
        hist = hist[hist > 0]
        return stats.entropy(hist, base=2)

    def process_criticality_transition(self, channel_idx: int = 0):
        """
        Segments data into 'Stress/Noise' and 'Meditation/Coherence' blocks,
        extracting mathematical metrics for portfolio display.
        """
        signal_data = self.raw_data[channel_idx]
        midpoint = len(signal_data) // 2
        
        block_chaos = signal_data[:midpoint]
        block_coherence = signal_data[midpoint:]
        
        beta_isolated = self.apply_butterworth_filter(block_chaos, 20.0, 30.0)
        hfd_chaos = self.calculate_higuchi_fractal_dimension(beta_isolated)
        entropy_chaos = self.calculate_shannon_entropy(beta_isolated)
        
        alpha_isolated = self.apply_butterworth_filter(block_coherence, 4.0, 12.0)
        hfd_coherent = self.calculate_higuchi_fractal_dimension(alpha_isolated)
        entropy_coherent = self.calculate_shannon_entropy(alpha_isolated)
        
        self._render_phase_space_plots(beta_isolated, alpha_isolated, hfd_chaos, 
                                      hfd_coherent, entropy_chaos, entropy_coherent)

    def _render_phase_space_plots(self, chaos_sig, coherent_sig, hfd_c, hfd_coh, ent_c, ent_coh):
        """
        Plots the 2D Phase Space Delay Embedding. Displays the structural transition 
        from high-entropy white noise to an organized geometric attractor.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Neural Criticality Map: Structural Shift of Information Fields', fontsize=14)
        
        # Plot Chaotic Phase Space (Stress/High Beta Noise)
        ax1.plot(chaos_sig[:-1], chaos_sig[1:], color='crimson', alpha=0.6, linewidth=0.5)
        ax1.set_title(f'State 1: Chaotic / Uncoupled\nHFD: {hfd_c:.3f} | Entropy: {ent_c:.3f} bits')
        ax1.set_xlabel('$x(t)$')
        ax1.set_ylabel('$x(t + \delta)$')
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # Plot Coherent Phase Space (Resonant Alpha/Theta Attractor)
        ax2.plot(coherent_sig[:-1], coherent_sig[1:], color='dodgerblue', alpha=0.7, linewidth=0.8)
        ax2.set_title(f'State 2: Phase-Locked / Coherent\nHFD: {hfd_coh:.3f} | Entropy: {ent_coh:.3f} bits')
        ax2.set_xlabel('$x(t)$')
        ax2.set_ylabel('$x(t + \delta)$')
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    pipeline = NeuralCriticalityPipeline()
    pipeline.load_synthetic_or_real_data(duration_seconds=12, num_channels=4)
    pipeline.process_criticality_transition(channel_idx=0)
