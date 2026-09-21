# Audio Signal Analyzer and Noise Reduction System Using FFT

An academic **Signals and Systems (SNS)** mini project demonstrating fundamental digital signal processing concepts:
- Discrete-time signal representation and time-domain analysis
- Frequency-domain transformation via Fast Fourier Transform (FFT)
- Signal degradation with controlled Additive White Gaussian Noise (AWGN)
- Digital filtering using Butterworth filters (Low-Pass, High-Pass, Band-Pass) with Second-Order Sections (SOS) and zero-phase distortion (`sosfiltfilt`)
- Objective Signal-to-Noise Ratio (SNR) evaluation before and after filtering
- Interactive desktop visualization built with PySide6 and Matplotlib

---

## Project Structure

```text
SNS_mini_project/
├── .venv/                      # Isolated Python virtual environment
├── audio_io.py                 # Audio loading, saving, downmixing & RMS calculations
├── signal_processing.py        # DSP core: FFT, AWGN, Butterworth filters, and SNR
├── gui.py                      # Desktop GUI layout and Matplotlib plotting integration
├── app.py                      # Application launcher
├── requirements.txt            # Pinned dependencies
├── README.md                   # Documentation & setup guide
├── sample_audio/               # Sample WAV audio files for demonstration
│   ├── generate_samples.py     # Script to generate synthetic test WAV files
│   ├── sine_440hz.wav          # Pure 440 Hz reference tone
│   ├── synthetic_voice.wav     # Multi-harmonic voice simulation
│   └── mixed_tone_300hz_3500hz.wav
├── output/                     # Saved processed audio files
└── tests/                      # Unit test suites
    ├── test_audio_io.py        # Audio I/O and validation tests
    └── test_signal_processing.py # DSP and filter tests
```

---

## Setup & Installation

### 1. Prerequisites
- Python 3.10+ installed on Windows, Linux, or macOS.

### 2. Create and Activate Virtual Environment
```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Generate Sample Audio Files
```powershell
python sample_audio/generate_samples.py
```

### 5. Run Unit Tests
```powershell
python -m unittest discover -s tests -p "test_*.py"
```

---

## DSP Concepts Covered

1. **Discrete-Time Representation ($x[n]$):**
   Audio is sampled at sampling frequency $f_s$ (e.g. 44,100 Hz). The time vector is $t[n] = n / f_s$.

2. **Fast Fourier Transform (FFT):**
   Converts the discrete time-domain signal $x[n]$ into its frequency-domain representation $X[k]$ using `scipy.fft.rfft` and `scipy.fft.rfftfreq` for real-valued signals, normalized to single-sided physical amplitude:
   $$\text{Magnitude}(f) = \frac{2 \cdot |X[k]|}{N}$$

3. **Additive White Gaussian Noise (AWGN):**
   Controlled noise generated using a normal distribution $\mathcal{N}(0, \sigma^2)$ calculated from the target SNR (dB) and input signal power:
   $$\sigma^2 = \frac{P_{\text{signal}}}{10^{\text{target\_SNR} / 10}}$$

4. **Butterworth Digital Filtering:**
   Maximally flat passband response designed in Second-Order Sections (SOS) to avoid numerical instability. Offline zero-phase forward-backward filtering (`scipy.signal.sosfiltfilt`) is employed to eliminate phase distortion.

5. **Signal-to-Noise Ratio (SNR):**
   Evaluated strictly when a ground-truth clean reference signal is available:
   $$\text{SNR}_{\text{dB}} = 10 \log_{10}\left( \frac{P_{\text{signal}}}{P_{\text{error}}} \right)$$
