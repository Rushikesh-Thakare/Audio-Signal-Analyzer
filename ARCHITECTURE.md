# System Architecture & DSP Pipeline Specification

This document details the architectural design, module relationships, and mathematical signal processing pipeline implemented in the **Audio Signal Analyzer and Noise Reduction System Using FFT**, including standalone integration for the **LINA Linux Voice Assistant**.

---

## 1. High-Level System Architecture

The application adopts a **decoupled Model-View-Controller (MVC)** architecture tailored for scientific computing, digital signal processing, and external voice assistant integration in Python:

- **Presentation Layer (`gui.py` & `app.py`):** PySide6 graphical widgets, background threads (`RecordWorker`), and embedded Matplotlib canvas figures (`FigureCanvasQTAgg`).
- **Controller / Dispatcher Layer (`gui.py` slots):** Handles user interactions, executes validation checks, manages dual evaluation models, and dispatches data.
- **DSP Engine / Core Model (`signal_processing.py`):** Pure numerical processing functions executing Short-Time Fourier Transforms (STFT), spectral gating, noise profile estimation, Butterworth IIR filter synthesis, and dual metric evaluations using NumPy and SciPy.
- **LINA Decoupled Interface (`signal_processing.denoise_audio`):** A lightweight, standalone Python entry point requiring zero GUI dependencies, directly callable by downstream automated speech recognition (ASR) pipelines.
- **Audio I/O Subsystem (`audio_io.py`):** Filesystem and hardware abstraction for reading/writing WAV files, stereo-to-mono downmixing, safe playback with `sounddevice`, and microphone capture.

---

## 2. Architectural Block Diagram

```mermaid
graph TD
    subgraph LINA_Integration ["External Linux Voice Assistant (LINA)"]
        LINA_Mic[Raw Microphone Stream] --> LINA_Hook["denoise_audio(audio, fs, strength)"]
        LINA_Hook --> LINA_ASR[Speech-to-Text / Intent Recognition Engine]
    end

    subgraph UI_Layer ["Desktop Presentation Layer (PySide6 + Matplotlib)"]
        A[app.py Launcher] --> B[MainWindow in gui.py]
        B --> C[Control Panels: Ingestion, Spectral Denoising, Playback, Metrics]
        B --> D[Visualizations: Comparison, Zoomed Waveforms, FFT, Spectrogram]
        B --> E[RecordWorker QThread]
    end

    subgraph Audio_Subsystem ["Audio I/O Subsystem (audio_io.py)"]
        F[load_audio] -->|Read WAV| FS[(File System)]
        G[save_audio] -->|Write Clamped WAV| FS
        H[to_mono] -->|Downmix Channel Averaging| F
        I[play_audio / stop_audio] -->|Sound Stream| SPK[Speakers / Headphones]
        J[record_audio] -->|Live Capture| MIC[Microphone Input]
        E -->|Worker Hook| J
    end

    subgraph DSP_Engine ["DSP Signal Processing Core (signal_processing.py)"]
        K[compute_time_axis / compute_rms]
        L[compute_fft / find_peak_frequency]
        M[detect_noise_segment / estimate_noise_profile]
        N[spectral_gate_denoise: STFT + Soft Wiener Mask + ISTFT]
        O[compute_spectrogram: Time-Frequency PSD]
        P[apply_filter: Butterworth SOS zero-phase]
        Q[calculate_unreferenced_metrics: Noise Attenuation & Speech Retention]
        R[calculate_snr: Ground-Truth Benchmark Evaluation]
    end

    %% Wiring connections
    C -->|Trigger Load| F
    C -->|Trigger Playback| I
    C -->|Trigger Save| G
    C -->|Trigger Mic Record| E
    
    F -->|Raw Signal Arrays| B
    B -->|Input Signal x[n]| K
    B -->|Signal x[n], fs| L
    B -->|Signal x[n], fs| O
    B -->|Signal x[n], fs, Range| M
    M -->|Noise Threshold T_k| N
    B -->|Signal, Strength, Threshold| N
    N -->|Cleaned Signal x_hat[n]| L
    N -->|Cleaned Signal x_hat[n]| O
    N -->|Cleaned Signal, Input| Q
    
    LINA_Hook --> N
    
    L -->|Freqs & Magnitudes| D
    K -->|Time Vector| D
    O -->|Spectrogram Matrices| D
    Q -->|Unreferenced Metrics| C
    R -->|Benchmark SNR dB| C
```

---

## 3. Module Breakdown & Responsibilities

### 3.1 `app.py` — Application Entry Point
- Initializes the Qt runtime environment (`QApplication`), sets application metadata, configures global typography (`Segoe UI`, 10pt standard font sizing), instantiates `MainWindow`, and starts the main event loop (`app.exec()`).

### 3.2 `gui.py` — Graphical User Interface & Visualizations
- Implements the complete desktop GUI window with high-contrast, modern academic styling.
- Houses sidebar control panels:
  - **1. Audio Ingestion & Metadata:** Load WAV, Record 3s Mic, Reset Workspace, and Metadata Grid (duration, samples, channels, RMS, peak frequencies).
  - **2. Noise Reduction & Denoising Engine:** Method selector (Speech Spectral Gating vs. Butterworth Filter), Reduction Strength slider (0% to 150%), Noise Profile strategy selector (Auto-detect quietest segment, leading 0.3s pause, or custom start/end time range), and Primary `Process & Clean Audio` button.
  - **3. Audio Playback & Export:** Independent playback for Input Audio, Cleaned Audio, and Benchmark Noisy Audio, plus non-overwriting WAV export.
  - **4. Quantitative Audio Evaluation:** Displays unreferenced measurements for real recordings and ground-truth metrics for benchmarks.
  - **5. Academic Lab Benchmark Tool:** Optional synthetic AWGN generator with target SNR spinbox.
- Embeds interactive Matplotlib figures via `MplCanvas`:
  - **Tab 0: Pipeline Comparison:** Side-by-side Input vs. Cleaned waveforms and FFT spectra.
  - **Tab 1: Input Audio:** High-detail view of input time waveform and magnitude spectrum with peak frequency marker.
  - **Tab 2: Cleaned Audio:** High-detail view of cleaned time waveform and magnitude spectrum.
  - **Tab 3: Time-Frequency Spectrogram:** Dual-row STFT spectrograms displaying power density heatmaps and vocal harmonic preservation.
  - **Tab 4: Benchmark Noisy Audio:** Detailed view of corrupted signal when AWGN is injected.
  - **Tab 5: Filter Frequency Response:** Bode magnitude response curves ($|H(f)|$) with $-3\text{ dB}$ cutoff markers.

### 3.3 `audio_io.py` — Audio File I/O & Hardware Abstraction
- Encapsulates file I/O and device interactions:
  - `load_audio(filepath)`: Reads 16/24/32-bit PCM and float WAV files, extracts metadata, and averages multi-channel stereo to mono via `to_mono()`.
  - `to_mono(audio_data)`: Converts stereo signals to 1D mono via arithmetic channel averaging: $x_{\text{mono}}[n] = \frac{1}{C}\sum_{c=1}^C x_c[n]$.
  - `calculate_rms(signal)`: Evaluates Root-Mean-Square signal amplitude.
  - `save_audio(filepath, audio_data, fs)`: Exports processed audio as 16-bit PCM WAV with hard clipping clamping strictly within $[-1.0, 1.0]$.
  - `play_audio(audio_data, fs)` & `stop_audio()`: Streams audio to output devices safely via `sounddevice` with non-crashing exception wrapping.
  - `is_microphone_available()` & `record_audio(duration, fs)`: Detects input recording hardware and captures calibrated 1D mono audio arrays.

### 3.4 `signal_processing.py` — Mathematical DSP Core
Pure scientific computation library implementing all discrete-time signal processing equations:
- `compute_time_axis(num_samples, fs)`: Uniform discrete time vector $t[n] = n / f_s$.
- `compute_signal_stats(signal)`: Time-domain statistical features (min, max, peak amplitude, DC offset, RMS).
- `compute_fft(signal, fs)`: Normalized one-sided real FFT (`rfft` and `rfftfreq`) scaled such that a pure sine wave of peak amplitude $A$ produces a magnitude peak of $A$.
- `find_peak_frequency(freqs, magnitudes)`: Detects dominant spectral peak above $20\text{ Hz}$.
- `detect_noise_segment(signal, fs, segment_duration_s)`: Sliding window minimum-energy search that intelligently locates quiet intervals without blindly assuming speech starts late.
- `estimate_noise_profile(signal, fs, start_s, end_s, fallback_duration_s)`: Evaluates frequency-dependent mean $\mu_{\text{noise}}(f)$ and standard deviation $\sigma_{\text{noise}}(f)$ magnitude spectra.
- `spectral_gate_denoise(signal, fs, strength, noise_profile, ...)`: STFT-based soft-knee Wiener masking with spectral floor and temporal smoothing.
- `denoise_audio(audio, fs, strength, ...)`: Decoupled, zero-GUI entry point for the LINA Linux voice assistant.
- `calculate_unreferenced_metrics(...)`: Objectively measures noise floor attenuation, speech energy retention, and overall RMS changes for real audio.
- `compute_spectrogram(signal, fs)`: STFT power spectral density calculation in decibels.
- `validate_filter_cutoffs(...)`, `apply_filter(...)`, `_apply_butterworth_sos(...)`: 4th-order digital Butterworth filters using Second-Order Sections (SOS) and zero-phase forward-backward filtering (`sosfiltfilt`).
- `compute_filter_response(...)`: Evaluates theoretical single-pass and effective zero-phase frequency responses via `scipy.signal.sosfreqz`.
- `add_white_gaussian_noise(...)` & `calculate_snr(...)`: Ground-truth AWGN generation and exact SNR evaluation for synthetic benchmark demonstrations.

---

## 4. End-to-End Processing Lifecycles

### 4.1 Primary Production Workflow (Noisy WAV / Live Mic)
```text
[Noisy Audio Input] (WAV / 3s Mic Capture)
       │
       ▼
[Stage 0: Audio Ingestion & Format Normalization]
  - Stereo downmixing: x_mono[n] = (L[n] + R[n]) / 2
  - Metadata: fs, N, duration, RMS
       │
       ├────────────────────────────────────────┐
       ▼                                        ▼
[Stage 1: Time Domain]                 [Stage 2: Frequency Domain & STFT]
  - t[n] = n / fs                        - FFT: X[k] = rfft(x[n])
  - RMS = sqrt(mean(x^2))                - Spectrogram: S_xx(f, t)
       │                                        │
       └──────────────────┬─────────────────────┘
                          │
                          ▼
               [Stage 3: Noise Profiling]
                 - Sliding window minimum-energy search OR user range
                 - Calculate mu_noise(f) and sigma_noise(f)
                 - Set threshold: T(f) = mu_noise(f) + 1.2 * sigma_noise(f)
                          │
                          ▼
               [Stage 4: Time-Frequency Spectral Gating]
                 - Soft Wiener Gain: G(f, t) = 1 / (1 + (alpha * T(f) / |Y(f, t)|)^2)
                 - Enforce spectral floor: G(f, t) >= beta (0.05 = -26 dB)
                 - Temporal inter-frame recursive smoothing
                 - Synthesize: x_hat[n] = ISTFT(G(f, t) * |Y(f, t)| * exp(j * phi(f, t)))
                 - Amplitude clamping to [-1.0, 1.0]
                          │
                          ▼
               [Stage 5: Objective Evaluation & Output]
                 - Noise Floor Attenuation: Delta_Noise = 20 * log10(RMS_noise_in / RMS_noise_out)
                 - Speech Energy Retention: Ratio = RMS_speech_out / RMS_speech_in
                 - Ground-Truth SNR: Honestly reported as N/A (Real Audio)
                 - Dual audio playback (sounddevice) & clamped WAV export
```

### 4.2 Academic Lab Benchmark Workflow (Synthetic Reference)
```text
[Known Clean WAV] -> [Add AWGN at Target SNR dB] -> [Noisy Signal y[n]]
       │                                                   │
       │                                                   ▼
       │                                       [Apply Denoising / Filtering]
       │                                                   │
       │                                                   ▼
       └─────────────────────────────────────────> [Calculate Ground-Truth SNR]
                                                     - SNR_before = 10 * log10(P_clean / P_noise)
                                                     - SNR_after  = 10 * log10(P_clean / P_residual)
                                                     - Delta SNR  = SNR_after - SNR_before
```

---

## 5. Numerical Stability & Academic Integrity Safeguards

1. **Second-Order Sections (SOS) Biquad Cascades:**
   Standard high-order IIR transfer functions in direct form suffer severe coefficient quantization instability. Cascaded second-order biquads guarantee numerical stability across all cutoff frequencies up to Nyquist.
2. **Zero-Phase Filtering (`sosfiltfilt`):**
   Forward-backward filtering cancels nonlinear phase distortions, producing identically zero phase response:
   $$\theta_{\text{net}}(\omega) = \theta(\omega) - \theta(\omega) = 0$$
3. **Soft-Knee Wiener Spectral Gating vs. Hard Gating:**
   Binary gating (zeroing bins below threshold) introduces high-frequency musical noise chirps. The continuous soft-knee Wiener mask combined with a $-26\text{ dB}$ spectral floor and temporal smoothing eliminates musical artifacts while preserving speech formants.
4. **Honest Metric Reporting:**
   The application never invents an SNR improvement figure when processing unreferenced real-world recordings. Ground-truth SNR is strictly reserved for synthetic experiments where the clean signal is known.
5. **Decoupled Architecture for LINA:**
   The core DSP routines in `signal_processing.py` are strictly independent of the GUI framework. LINA voice assistant scripts can import `denoise_audio` directly without launching Qt or Matplotlib.
