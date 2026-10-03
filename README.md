# Audio Signal Analyzer and Noise Reduction System Using FFT

An academic **Signals and Systems (SNS)** project demonstrating core digital signal processing (DSP) principles through an interactive desktop application.

Developed as a standalone audio preprocessing engine for the **LINA Linux Voice Assistant** and an educational laboratory workstation for Signals & Systems engineering students.

---

## Key Capabilities & Core Workflows

### 1. Primary Workflow: Noisy Audio Ingestion & Speech Denoising
- **Input:** Real-world noisy WAV recordings or 3-second live microphone capture.
- **Speech-Preserving Spectral Gating:** Applies Short-Time Fourier Transform (STFT) with soft-knee Wiener-type gain masking to suppress ambient stationary background noise (PC fan noise, HVAC hum, room rumble, electrical hiss) while preserving vocal formants and speech intelligibility.
- **Intelligent Noise Profiling:** Does **not** blindly assume the initial 0.3 seconds is noise (which would corrupt recordings where the user speaks immediately). Instead, scans the recording using sliding windows to detect the quietest pause, allows manual user range selection, or samples leading silence.
- **Dual Playback & Clamped Export:** Listen to the input audio vs. the cleaned audio side-by-side using `sounddevice`, and export 16-bit PCM WAV safely clamped to $[-1.0, 1.0]$ without digital clipping.

### 2. Standalone LINA Voice Assistant Preprocessor
- Provides a completely decoupled, zero-GUI Python function:
  ```python
  from signal_processing import denoise_audio

  # Accepts 1D numpy array and sampling rate; returns cleaned 1D audio
  clean_audio = denoise_audio(raw_mic_stream, fs=16000, strength=0.75)
  ```
- 100% offline, lightweight, deterministic, and requires no GPU or external cloud/neural dependencies.

### 3. Classical Digital Filtering (Butterworth SOS)
- 4th-order Low-Pass, High-Pass, and Band-Pass IIR filters implemented in **Second-Order Sections (SOS)** biquad cascade format for high numerical stability.
- Bidirectional zero-phase filtering (`scipy.signal.sosfiltfilt`) prevents phase lag and temporal dispersion.
- Interactive Bode magnitude frequency response $|H(f)|$ tab displaying single-pass and effective zero-phase curves with $-3\text{ dB}$ cutoff markers.

### 4. Honest Scientific Evaluation & Dual Metric Engine
- **Real Audio (Unreferenced):** A real-world recording has no clean ground truth. Rather than fabricating or inventing an SNR number, the system honestly reports:
  - **Noise Floor Attenuation ($\text{dB}$):** Measured power reduction on stationary noise pauses.
  - **Speech Energy Retention ($\%$):** Ratio of active speech energy preserved (proves speech was not attenuated away).
  - **Overall RMS Level Change ($\text{dB}$)**.
  - **Ground-Truth SNR:** Honestly labelled as `N/A (Real Audio — No clean reference)`.
- **Academic Lab Benchmark Tool:** Optional feature to inject controlled Additive White Gaussian Noise (AWGN) at a chosen target SNR into a clean reference signal to calculate and demonstrate mathematically exact ground-truth $\text{SNR}_{\text{before}}$, $\text{SNR}_{\text{after}}$, and $\Delta\text{SNR}$ improvement.

### 5. Multi-Domain Visualizations (PySide6 & Matplotlib)
- **Pipeline Overview:** Side-by-side time waveforms and FFT magnitude spectra.
- **Stage Detail Views:** Zoomed time-domain signals and calibrated one-sided magnitude spectra with dominant peak frequency annotations.
- **Time-Frequency Spectrograms:** High-resolution STFT power spectral density heatmaps displaying vocal harmonic tracks and noise floor reduction.
- **Bode Response:** Theoretical transfer function plots.

---

## Project Structure

```text
SNS_mini_project/
├── .venv/                      # Python virtual environment
├── app.py                      # Primary desktop application entry point
├── gui.py                      # PySide6 desktop GUI & Matplotlib plotting integration
├── audio_io.py                 # Audio I/O, downmixing, RMS, playback, microphone
├── signal_processing.py        # DSP core: STFT spectral gating, FFT, AWGN, Butterworth SOS, SNR
├── requirements.txt            # Pinned dependencies
├── README.md                   # Project overview and instructions
├── ARCHITECTURE.md             # System architecture and mathematical DSP pipeline
├── DEMO_SCRIPT.md              # Demonstration walkthrough and viva demo guide
├── VIVA.md                     # 25 Signals & Systems viva exam questions & detailed answers
├── sample_audio/               # Sample WAV audio files
│   ├── generate_samples.py     # Generator for synthetic test audio
│   ├── sine_440hz.wav          # 440 Hz reference tone
│   ├── synthetic_voice.wav     # Multi-harmonic vocal formant simulation
│   └── mixed_tone_300hz_3500hz.wav # Dual-tone signal for filter demo
├── output/                     # Saved processed audio files
└── tests/                      # Automated test suite (60 unit & integration tests)
    ├── test_audio_io.py        # Audio I/O, downmixing, clipping, and playback tests
    ├── test_signal_processing.py # Discrete time, FFT, Butterworth SOS, and SNR tests
    ├── test_denoising.py       # Speech spectral gating, noise profiling, and LINA tests
    ├── test_gui.py             # Headless PySide6 GUI workflow & widget interaction tests
    ├── test_e2e_audit.py       # End-to-end DSP pipeline and robustness audit tests
    └── verify_launch.py        # App initialization and smoke test script
```

---

## Setup & Installation

### 1. Prerequisites
- Python 3.10 or higher installed on Windows, Linux, or macOS.

### 2. Create and Activate Virtual Environment

```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

Core dependencies in `requirements.txt`:
- `numpy>=1.23.0`
- `scipy>=1.9.0`
- `matplotlib>=3.6.0`
- `soundfile>=0.12.0`
- `sounddevice>=0.4.6`
- `PySide6>=6.5.0`

### 4. Generate Sample Audio Files

```powershell
python sample_audio/generate_samples.py
```

Generates three test signals in `sample_audio/`:
- `sine_440hz.wav`: Pure 440 Hz reference tone ($f_s = 44100\text{ Hz}$).
- `synthetic_voice.wav`: 150 Hz fundamental frequency with vocal formants.
- `mixed_tone_300hz_3500hz.wav`: Low-frequency 300 Hz tone mixed with high-frequency 3500 Hz noise tone.

---

## Run Commands

### 1. Launch the Desktop Application

```powershell
python app.py
```
*(Or `.\.venv\Scripts\python.exe app.py`)*

### 2. Run the Full Automated Test Suite (60 Tests)

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

### 3. Verify GUI Clean Launch and Shutdown

```powershell
python tests/verify_launch.py
```

---

## Step-by-Step Usage Guide

### Primary Workflow: Cleaning Noisy Audio
1. **Load Noisy Audio:**
   - Click **Load Audio WAV** and choose a recorded audio file (e.g., `sample_audio/synthetic_voice.wav` or your own noisy voice recording).
   - Alternatively, click **Record (3s)** to capture audio directly from your microphone.
   - The waveform, FFT spectrum, and Spectrogram appear in the visualization panel.
2. **Configure Noise Reduction:**
   - Leave **Method** set to `Speech Spectral Denoising (STFT Wiener)`.
   - Set the **Strength** slider (default 75%).
   - Select **Noise Sample**: Choose `Auto-Detect Quietest Segment` (default), `Leading Segment (First 0.3s)`, or `Custom Range`.
3. **Process Audio:**
   - Click **✨ Process & Clean Audio**.
   - The cleaned waveform, cleaned spectrum, and cleaned spectrogram update immediately.
   - Check the **Quantitative Evaluation** card: see measured **Noise Floor Attenuation** (e.g. 8–18 dB) and **Speech Energy Retention** (e.g. 92–98%).
4. **Listen & Compare:**
   - Click **▶ Play Input Audio** to hear the original noisy track.
   - Click **▶ Play Cleaned Audio** to hear the restored, denoised result.
5. **Export:**
   - Click **💾 Save Cleaned WAV** to export the result to the `output/` folder. Overwrite protection prevents accidental overwriting of the input file.

### Academic Demonstration: Synthetic AWGN Benchmark Mode
1. Load a clean audio file (e.g., `sample_audio/sine_440hz.wav`).
2. Scroll to Section 5: **Academic Lab Benchmark Tool**.
3. Choose a target SNR (e.g., `10.0 dB`) and click **⚡ Inject Known AWGN (Lab Test)**.
4. The system stores the original audio as a ground-truth reference and corrupts the input.
5. Click **✨ Process & Clean Audio**.
6. The metrics card now displays exact ground-truth values:
   - `SNR Before: 10.02 dB`
   - `SNR After: 18.45 dB`
   - `SNR Improvement: +8.43 dB`

---

## DSP Theory & Mathematical Foundations

### 1. Discrete-Time Fourier Transform (STFT)
Audio $x[n]$ is segmented into windowed frames using a Hann window $w[m]$:
$$X(k, m) = \sum_{n=0}^{N-1} x[n + mH] w[n] e^{-j 2\pi k n / N}$$
where $H$ is the hop size (75% overlap) and $N$ is the FFT segment length.

### 2. Wiener-Type Soft Spectral Mask
For frequency bin $k$ and time frame $m$, the gain mask is:
$$G(k, m) = \max\left(\beta, \frac{1}{1 + \left(\frac{\alpha \cdot T_{\text{noise}}[k]}{|Y(k, m)| + \epsilon}\right)^2}\right)$$
where:
- $T_{\text{noise}}[k] = \mu_{\text{noise}}[k] + 1.2 \cdot \sigma_{\text{noise}}[k]$ is the noise threshold.
- $\alpha$ is the user-controlled **reduction strength** (0.0 to 1.5).
- $\beta = 0.05$ (-26 dB) is the **spectral floor** that prevents musical noise chirps.

### 3. Zero-Phase Inverse STFT (ISTFT) Synthesis
$$\hat{x}[n] = \text{ISTFT}\left( G(k, m) \cdot |Y(k, m)| \cdot e^{j \phi(k, m)} \right)$$
The phase spectrum $\phi(k, m)$ is fully preserved, preventing temporal dispersion.

---

## System Limitations & Design Boundaries

1. **Stationary vs. Non-Stationary Noise:**
   - Spectral gating assumes the noise spectrum is reasonably stationary over the duration of the clip (e.g., steady fan hum, computer background noise, HVAC rumble).
   - Highly transient or sudden impulsive noises (door slams, claps, background babble speech) require multi-microphone spatial beamforming or deep recurrent models.
2. **Extreme Negative SNR ($< -10\text{ dB}$):**
   - If the speech signal is substantially weaker than the noise floor across all frequency bins, spectral gating will attenuate speech together with the noise floor.
3. **Acoustic Reverberation:**
   - Spectral gating suppresses additive noise but does not deconvolve room reverberation (echoes caused by wall reflections).
4. **Causality & Real-Time Latency:**
   - The STFT implementation uses bidirectional framing and temporal smoothing. In a live streaming voice assistant pipeline, causal asymmetric windows with fixed algorithmic delay (typically 20–40 ms) must be employed.

---

## Author & Project Information
- **Course:** Signals and Systems (SNS) Mini Project
- **Target Integration:** LINA Linux Voice Assistant Audio Front-End
- **Frameworks:** Python, SciPy, NumPy, Matplotlib, PySide6 (Qt)
