# System Architecture & DSP Pipeline Specification

This document details the architectural design, module relationships, and mathematical signal processing pipeline implemented in the **Audio Signal Analyzer and Noise Reduction System Using FFT**.

---

## 1. High-Level System Architecture

The application adopts a clean **Model-View-Controller (MVC)** inspired architecture tailored for scientific computing and digital signal processing in Python:

- **View & Interaction Layer (`gui.py` & `app.py`):** PySide6 (Qt) graphical widgets, asynchronous worker threads (`QThread`), and embedded Matplotlib canvas figures.
- **Controller / Dispatcher Layer (`gui.py` slots):** Handles user interactions, executes validation checks, and dispatches data between UI controls and the DSP engine.
- **DSP Engine / Core Model (`signal_processing.py`):** Pure numerical processing functions executing Fourier transforms, Gaussian noise generation, IIR filter synthesis, and SNR evaluations using NumPy and SciPy.
- **Audio I/O Subsystem (`audio_io.py`):** Hardware and filesystem abstraction for reading/writing WAV files, stereo downmixing, safe playback with `sounddevice`, and optional microphone capture.

---

## 2. Architectural Block Diagram

```mermaid
graph TD
    subgraph UI_Layer ["Presentation & Interaction Layer (PySide6 + Matplotlib)"]
        A[app.py Entry Point] --> B[MainWindow in gui.py]
        B --> C[Control Panels: Load, Noise, Filter, Play, Reset]
        B --> D[Matplotlib Canvases: 3-Stage Overview & Detailed Tabs]
        B --> E[RecordWorker QThread]
    end

    subgraph Audio_Subsystem ["Audio I/O Subsystem (audio_io.py)"]
        F[load_audio] -->|Read WAV| FS[(File System)]
        G[save_audio] -->|Write Clamped WAV| FS
        H[to_mono] -->|Downmix| F
        I[play_audio / stop_audio] -->|Sound Stream| SPK[Speakers / Headphones]
        J[record_audio] -->|Live Capture| MIC[Microphone Input]
        E -->|Worker Hook| J
    end

    subgraph DSP_Engine ["DSP Signal Processing Core (signal_processing.py)"]
        K[compute_time_axis]
        L[compute_fft / find_peak_frequency]
        M[add_white_gaussian_noise]
        N[design_butterworth_filter]
        O[apply_filter via sosfiltfilt]
        P[compute_filter_response via sosfreqz]
        Q[calculate_snr Ground Truth]
    end

    %% Wiring connections
    C -->|Trigger Load| F
    C -->|Trigger Playback| I
    C -->|Trigger Save| G
    C -->|Trigger Mic Record| E
    
    F -->|Raw Signal Arrays| B
    B -->|Clean Signal x[n]| K
    B -->|Signal x[n], fs| L
    B -->|Clean x[n], Target SNR| M
    M -->|Noisy Signal y[n]| L
    B -->|Filter Type, Cutoffs, fs| N
    N -->|SOS Matrix| P
    N -->|SOS Matrix, Noisy y[n]| O
    O -->|Filtered Signal x_hat[n]| L
    B -->|Clean x[n], Noisy / Filtered| Q
    
    L -->|Freqs & Magnitudes| D
    K -->|Time Vector| D
    P -->|Bode Mag & Phase| D
    Q -->|SNR Values (dB)| C
```

---

## 3. Module Breakdown & Responsibilities

### 3.1 `app.py` — Application Entry Point
- **Role:** Initializes the Qt runtime environment (`QApplication`), sets global typography (Segoe UI, 10pt standard font sizing), instantiates `MainWindow`, and starts the main event loop (`app.exec()`).
- **Dependencies:** `PySide6.QtWidgets`, `PySide6.QtGui`, `gui.MainWindow`.

### 3.2 `gui.py` — Graphical User Interface & Visualizations
- **Role:** Implements the complete desktop GUI window.
  - Houses the sidebar control layout: File Loading, Microphone Recording, Metadata Grid, Noise Generator, Butterworth Filter Controls, SNR Metrics Display, Audio Playback Controls, and System Reset.
  - Embeds interactive Matplotlib figures via `FigureCanvasQTAgg`:
    - **Tab 0: Pipeline Overview:** $3 \times 2$ grid plotting Original vs. Noisy vs. Filtered waveforms and frequency spectra side-by-side.
    - **Tab 1: Original Signal:** High-detail view of clean time-domain waveform and magnitude spectrum with peak frequency marker.
    - **Tab 2: Noisy Signal:** High-detail view of corrupted waveform and elevated noise floor.
    - **Tab 3: Filtered Signal:** High-detail view of restored waveform and cleaned spectrum.
    - **Tab 4: Filter Response:** Bode magnitude (in dB) and phase (in degrees) plots showing the $-3\text{ dB}$ cutoff point and attenuation slopes.
  - Implements `RecordWorker(QThread)` for non-blocking 3-second live microphone capture.

### 3.3 `audio_io.py` — Audio File I/O & Hardware Abstraction
- **Role:** Encapsulates all interactions with audio files and hardware devices.
  - `load_audio(filepath)`: Reads WAV files, extracts sample rate, channel count, duration, raw data, and automatic downmixed mono data.
  - `to_mono(audio_data)`: Converts stereo signals to mono via mathematical averaging $\frac{1}{2}(L + R)$.
  - `calculate_rms(signal)`: Evaluates Root-Mean-Square signal amplitude.
  - `save_audio(filepath, audio_data, fs)`: Exports processed audio with hard clipping enforcement strictly within $[-1.0, 1.0]$.
  - `play_audio(audio_data, fs)` & `stop_audio()`: Streams audio to output devices safely via `sounddevice` with non-crashing exception wrapping.
  - `is_microphone_available()` & `record_audio(duration, fs)`: Detects input recording hardware and captures calibrated 1D mono audio arrays.

### 3.4 `signal_processing.py` — Mathematical DSP Core
- **Role:** Pure scientific computation library implementing all core Signals and Systems equations:
  - `compute_time_axis(num_samples, fs)`: Generates uniform discrete time vector $t[n] = n / f_s$.
  - `compute_signal_stats(signal)`: Extracts statistical features (min, max, peak amplitude, DC offset, RMS).
  - `compute_fft(signal, fs)`: Calculates normalized one-sided magnitude spectrum using `np.fft.rfft` and `np.fft.rfftfreq`.
  - `find_peak_frequency(freqs, magnitudes)`: Detects dominant spectral peak above $0\text{ Hz}$.
  - `add_white_gaussian_noise(signal, target_snr_db, seed)`: Synthesizes AWGN scaled to exact signal power and target SNR.
  - `validate_cutoff_frequency(cutoff, fs, filter_type)`: Validates mathematical constraints ($0 < f_c < f_s/2$).
  - `design_butterworth_filter(filter_type, cutoff, fs, order=4)`: Generates Second-Order Sections (SOS) biquad matrix using `scipy.signal.butter`.
  - `apply_filter(signal, filter_type, cutoff, fs, order=4)`: Applies zero-phase digital filtering using `scipy.signal.sosfiltfilt`.
  - `compute_filter_response(filter_type, cutoff, fs, order=4, num_points=1024)`: Computes complex frequency response $H(e^{j\omega})$ via `scipy.signal.sosfreqz`.
  - `calculate_snr(clean_signal, noisy_or_processed_signal)`: Computes ground-truth Signal-to-Noise Ratio in decibels.

---

## 4. Detailed End-to-End DSP Pipeline

The processing lifecycle follows five mathematically rigorous stages:

```text
[WAV File / Mic]
       │
       ▼
[Stage 0: Audio Ingestion]
  - Stereo downmix: x_mono[n] = (L[n] + R[n]) / 2
  - Metadata: fs, N, duration, RMS
       │
       ├────────────────────────────────────────┐
       ▼                                        ▼
[Stage 1: Time Domain]                 [Stage 2: Frequency Domain]
  - t[n] = n / fs                        - X[k] = rfft(x[n])
  - Min, Max, Peak                       - f[k] = k * fs / N
  - RMS = sqrt(mean(x^2))                - Mag[k] = 2 * |X[k]| / N
                                         - Peak Freq = argmax(Mag[k > 0])
       │                                        │
       └──────────────────┬─────────────────────┘
                          │
                          ▼
               [Stage 3: Noise Injection (AWGN)]
                 - P_x = mean(x[n]^2)
                 - sigma^2 = P_x / 10^(target_SNR / 10)
                 - w[n] ~ N(0, sigma^2)
                 - y[n] = x[n] + w[n]
                 - Measure SNR_before = 10 * log10(P_x / P_noise)
                          │
                          ▼
               [Stage 4: Butterworth SOS Filtering]
                 - Validate 0 < f_c < fs / 2
                 - Design 4th-order SOS matrix: scipy.signal.butter(..., output='sos')
                 - Zero-phase filtering: x_hat[n] = scipy.signal.sosfiltfilt(sos, y[n])
                          │
                          ▼
               [Stage 5: Objective Evaluation & Output]
                 - Error signal: e[n] = x_hat[n] - x[n]
                 - SNR_after = 10 * log10(P_x / mean(e[n]^2))
                 - Delta SNR = SNR_after - SNR_before
                 - Safe WAV export with [-1.0, 1.0] clamping
```

---

## 5. Numerical Stability & Academic Integrity Safeguards

1. **Second-Order Sections (SOS) Biquad Cascades:**
   Standard high-order IIR polynomials in $(b, a)$ format suffer severe coefficient quantization noise. Expressing the filter as a cascade of 2nd-order sections guarantees numerical stability across all valid cutoff frequencies.
2. **Zero-Phase Forward-Backward Filtering (`sosfiltfilt`):**
   Standard causal filtering introduces non-linear phase distortion and group delays. Zero-phase processing eliminates phase shift entirely:
   $$\theta_{\text{net}}(\omega) = \theta(\omega) - \theta(\omega) = 0$$
   This prevents temporal dispersion and preserves transient alignments.
3. **Strict Array Immutability:**
   Adding noise or applying a filter allocates and returns a completely new NumPy array. The original `clean_signal` is never mutated in memory.
4. **Honest SNR Measurement:**
   SNR is only computed when an authentic clean reference $x[n]$ is available in memory. External unlabelled WAV files without injected noise honestly display `SNR: N/A`.
5. **Robust Exception Wrapping on Audio Hardware:**
   If a host machine lacks audio output speakers or microphone hardware, `audio_io.py` catches `sounddevice.PortAudioError` and displays informative warnings without crashing the application.
