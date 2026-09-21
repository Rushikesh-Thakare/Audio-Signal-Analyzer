# Audio Signal Analyzer and Noise Reduction System Using FFT

An academic **Signals and Systems (SNS)** mini project demonstrating core digital signal processing (DSP) principles through an interactive desktop application.

Built strictly using classical linear time-invariant (LTI) signal processing methods in Python:
- **Discrete-time signal representation and time-domain analysis**
- **Frequency-domain transformation via Fast Fourier Transform (FFT)**
- **Controlled signal degradation using Additive White Gaussian Noise (AWGN)**
- **Digital Butterworth filtering (Low-Pass, High-Pass, Band-Pass) with Second-Order Sections (SOS)**
- **Zero-phase forward-backward filtering (`sosfiltfilt`) to prevent phase distortion**
- **Objective Signal-to-Noise Ratio (SNR) evaluation before and after filtering**
- **Filter frequency response visualization (Bode magnitude & phase)**
- **Interactive desktop GUI built with PySide6 and Matplotlib**

---

## Features

1. **WAV Audio Loading & Format Handling:**
   - Supports 16-bit, 24-bit, 32-bit PCM and float WAV files via `soundfile`.
   - Automatic downmixing of multi-channel stereo audio to mono via arithmetic channel averaging:
     $$x_{\text{mono}}[n] = \frac{x_L[n] + x_R[n]}{2}$$
   - Amplitude clipping protection on export to keep output strictly within $[-1.0, 1.0]$.

2. **Signal Analysis & Peak Frequency Detection:**
   - Time axis generation: $t[n] = n / f_s$.
   - Comprehensive statistical extraction: Minimum, Maximum, Peak Amplitude, DC Offset, and Root-Mean-Square (RMS) Energy:
     $$\text{RMS} = \sqrt{\frac{1}{N}\sum_{n=0}^{N-1} |x[n]|^2}$$
   - Normalized one-sided real FFT (`rfft` and `rfftfreq`) scaled such that a pure sine wave of peak amplitude $A$ produces a magnitude peak of $A$.
   - Automatic dominant peak frequency detection with spectral indicator markers.

3. **Controlled Noise Generation (AWGN):**
   - Synthesizes zero-mean Additive White Gaussian Noise $w[n] \sim \mathcal{N}(0, \sigma^2)$.
   - Noise variance calculated from input signal power $P_x$ and user-specified target SNR (dB):
     $$\sigma^2 = \frac{P_x}{10^{\text{target\_SNR}_{\text{dB}} / 10}}$$
   - Strict immutability: clean signal array is never mutated; noisy signal is allocated separately.
   - Configurable pseudo-random seed for deterministic laboratory repeatability.

4. **Digital Butterworth Filtering:**
   - Implements 4th-order Low-Pass, High-Pass, and Band-Pass IIR Butterworth filters.
   - Designed using Second-Order Sections (SOS) matrix format for maximum numerical stability.
   - Forward-backward zero-phase filtering (`scipy.signal.sosfiltfilt`) eliminates group delay and phase lag.
   - Strict mathematical boundary validation against the Nyquist frequency ($f_c < f_s / 2$) and cutoff ordering ($f_{\text{low}} < f_{\text{high}}$).

5. **Filter Frequency Response (Bode Plot):**
   - Dedicated interactive tab plotting filter magnitude response $|H(e^{j\omega})|$ in decibels (dB) and phase response in degrees.
   - Displays $-3\text{ dB}$ half-power cutoff lines and passband/stopband behavior.

6. **Ground-Truth SNR Evaluation:**
   - Mathematically accurate Signal-to-Noise Ratio computed using clean ground-truth reference $x[n]$ and error signal $e[n] = y[n] - x[n]$:
     $$\text{SNR}_{\text{dB}} = 10 \log_{10}\left( \frac{\sum_{n=0}^{N-1} x[n]^2}{\sum_{n=0}^{N-1} (y[n] - x[n])^2} \right)$$
   - Evaluates **SNR Before Filtering**, **SNR After Filtering**, and **SNR Improvement ($\Delta$SNR)**.
   - Academic integrity: never fabricates SNR when a clean reference is absent.

7. **Interactive GUI & Audio Playback:**
   - Clean, modern layout built with **PySide6 (Qt)** and integrated **Matplotlib** canvases.
   - Dual inspection modes: Full 3-Stage Pipeline Overview (Original vs. Noisy vs. Filtered) and high-detail individual tabs.
   - Safe audio playback controls (`Play Original`, `Play Noisy`, `Play Filtered`, `Stop Audio`) powered by `sounddevice`.
   - Export Filtered WAV dialog.
   - Reset button to clear all loaded data and return to initial state.
   - Optional 3-second live microphone recording with non-blocking `QThread` worker and graceful hardware fallback.

---

## Project Structure

```text
SNS_mini_project/
├── .venv/                      # Python virtual environment
├── app.py                      # Application launch script
├── gui.py                      # PySide6 desktop GUI and plotting integration
├── audio_io.py                 # Audio I/O, downmixing, RMS, playback, microphone
├── signal_processing.py        # DSP core: FFT, AWGN, Butterworth SOS filters, SNR
├── requirements.txt            # Pinned dependencies
├── README.md                   # Project overview and instructions
├── ARCHITECTURE.md             # System architecture and DSP pipeline
├── DEMO_SCRIPT.md              # Demonstration walkthrough and script
├── VIVA.md                     # 20 Signals & Systems viva exam questions & answers
├── sample_audio/               # Sample WAV audio files
│   ├── generate_samples.py     # Generator for synthetic test audio
│   ├── sine_440hz.wav          # 440 Hz pure tone (tuning A)
│   ├── synthetic_voice.wav     # Multi-harmonic speech simulation
│   └── mixed_tone_300hz_3500hz.wav # Dual-tone signal for filter demo
├── output/                     # Saved processed audio files
└── tests/                      # Automated test suite (47 unit/integration tests)
    ├── test_audio_io.py        # Audio I/O, microphone, and playback tests
    ├── test_signal_processing.py # DSP math, FFT, noise, filter, and SNR tests
    ├── test_gui.py             # Headless PySide6 GUI workflow tests
    ├── test_e2e_audit.py       # End-to-end DSP pipeline audit tests
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

Pinned dependencies in `requirements.txt`:
- `numpy==2.2.3`
- `scipy==1.15.2`
- `matplotlib==3.10.1`
- `soundfile==0.13.1`
- `sounddevice==0.5.1`
- `PySide6==6.8.2.1`

### 4. Generate Sample Audio Files

```powershell
python sample_audio/generate_samples.py
```

This generates three standardized test signals in `sample_audio/`:
- `sine_440hz.wav`: Pure 440 Hz reference tone ($f_s = 44100\text{ Hz}$, duration $2.0\text{s}$).
- `synthetic_voice.wav`: 150 Hz fundamental frequency with 5 voice formants.
- `mixed_tone_300hz_3500hz.wav`: Low-frequency 300 Hz tone mixed with high-frequency 3500 Hz noise tone ($f_s = 16000\text{ Hz}$, duration $2.0\text{s}$).

---

## Run Commands

### 1. Launch the Desktop Application

```powershell
python app.py
```
*(Or `.\.venv\Scripts\python.exe app.py`)*

### 2. Run the Full Test Suite

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

### 3. Verify GUI Clean Launch and Shutdown

```powershell
python tests/verify_launch.py
```

---

## Step-by-Step Usage Guide

1. **Load an Audio File:**
   - Click **Load WAV File** and select `sample_audio/mixed_tone_300hz_3500hz.wav`.
   - The metadata panel updates with sample rate, duration, channel count, RMS energy, and dominant peak frequency.
   - The time-domain waveform and FFT spectrum appear in the **Pipeline Overview** and **Original Signal** tabs.

2. **Add Controlled Noise:**
   - In the **Noise Generator** card, set the target SNR (e.g., `10.0 dB`).
   - Click **Add Noise**.
   - The noisy waveform and flat noise floor on the FFT spectrum are rendered.
   - The **SNR Before** displays `~10.0 dB`.
   - Click **Play Noisy** to listen to the degraded audio.

3. **Configure & Inspect Filter:**
   - Select **Filter Type**: Choose `Low-pass`.
   - Set **Cutoff Frequency (Hz)**: Enter `1000.0` (well above 300 Hz and below 3500 Hz).
   - Switch to the **Filter Response** tab to inspect the Bode magnitude plot and verify attenuation at 3500 Hz.

4. **Apply Filter:**
   - Click **Apply Filter**.
   - The filtered waveform and cleaned FFT spectrum appear.
   - The high-frequency 3500 Hz tone and high-frequency noise floor are visibly attenuated.
   - **SNR After** and **SNR Improvement ($\Delta$SNR)** update showing positive decibel improvement.

5. **Listen & Export:**
   - Click **Play Filtered** to hear the restored signal.
   - Click **Save Filtered WAV** to export the result into `output/`.

6. **Reset:**
   - Click **Reset All** to clear all waveforms and return to a clean initial state.

---

## Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `Cutoff frequency must be less than Nyquist limit (fs / 2)` | The specified filter cutoff exceeds half of the sampling rate. | Ensure $f_{\text{cutoff}} < f_s / 2$. The GUI automatically constrains the maximum input to $(f_s/2 - 1)\text{ Hz}$. |
| `Low cutoff must be strictly less than high cutoff` | For band-pass filter, the lower cutoff was entered greater than or equal to the higher cutoff. | Set $f_{\text{low}} < f_{\text{high}}$ (e.g., $300\text{ Hz}$ and $3000\text{ Hz}$). |
| `[Audio Playback Notice] Could not play audio` | No default audio output device (speakers/headphones) detected by the OS. | Connect an output device. The application catches this safely and displays a warning without crashing. |
| `No audio recording device (microphone) detected` | No microphone connected or permission denied. | The app displays an alert and prompts you to use `Load WAV File` instead. |
| Headless test error: `Could not connect to display` | Running on a headless CI/Linux server without an X11 display. | Run with `QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests`. |

---

## Academic DSP Limitations

1. **LTI Filtering vs. In-Band Noise:**
   - Butterworth filters are linear time-invariant (LTI) frequency-selective filters. They attenuate noise in stopbands, but cannot separate noise that occupies the same frequency band as the desired signal.
   - Non-stationary or in-band noise reduction requires adaptive filters (e.g., LMS/RLS) or spectral subtraction.

2. **Zero-Phase Offline Processing:**
   - Zero-phase filtering (`sosfiltfilt`) processes the signal forward then backward. This requires the entire signal to be held in memory, making it suitable for recorded audio files, but not causally realizable in real-time streaming audio without latency.

3. **Ground-Truth Dependency for SNR:**
   - Exact mathematical SNR calculation requires access to the uncorrupted clean reference signal $x[n]$. In real-world field recordings where only noisy audio is captured, true SNR cannot be computed directly and must be estimated via statistical Voice Activity Detection (VAD).

4. **Butterworth Roll-Off Slope:**
   - The 4th-order Butterworth filter exhibits a smooth roll-off slope of $-24\text{ dB/octave}$ ($-80\text{ dB/decade}$). It does not possess an infinite ("brick-wall") transition band; frequency components immediately adjacent to the cutoff frequency experience partial attenuation.
