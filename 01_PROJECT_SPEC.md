# Audio Signal Analyzer and Noise Reduction System Using FFT

## Project Title
**Audio Signal Analyzer and Noise Reduction System Using FFT**

## Subject
**Signals and Systems (SNS)**

## Goal
Build a simple, reliable Python desktop application that loads or records audio, analyzes it in the time and frequency domains, adds controlled noise, reduces unwanted components using digital filters, compares the results, and saves/plays the processed audio.

## SNS Concepts Demonstrated
- Sampling and discrete-time signals
- Time-domain waveform
- Frequency-domain representation
- Fourier Transform / FFT
- Noise
- Low-pass filter
- High-pass filter
- Band-pass filter
- Cutoff frequency
- Signal reconstruction / processed output
- Signal-to-Noise Ratio (SNR)

## Important Scope
This is an SNS/DSP project, not an AI project. Do not add ML, LLM APIs, databases, cloud services, ESP32/Arduino, or custom hardware unless explicitly requested later.

## Required Features
1. Load a WAV file.
2. Show sampling rate, samples, duration, channels and RMS.
3. Plot original waveform.
4. Calculate and plot FFT magnitude spectrum.
5. Add controlled white Gaussian noise.
6. Plot noisy waveform and noisy FFT.
7. Select low-pass, high-pass or band-pass Butterworth filter.
8. Configure cutoff frequency/frequencies.
9. Apply the filter with valid cutoff checking.
10. Plot filtered waveform and filtered FFT.
11. Show SNR before/after filtering only when a clean reference is known.
12. Play original/noisy/filtered audio when the local audio device supports it.
13. Save filtered WAV.
14. Handle bad files, invalid cutoff values and unavailable microphone/audio devices gracefully.
15. Work offline after dependencies are installed.

## Recommended Stack
- Python 3
- NumPy
- SciPy
- Matplotlib
- PySide6
- SoundFile
- SoundDevice (optional)

## Architecture

```text
WAV Upload / Optional Microphone
              |
              v
      Audio Input + Validation
              |
              v
       Original Waveform
              |
              +------------------+
              |                  |
              v                  v
        Add White Noise         FFT
              |                  |
              v                  v
         Noisy Signal       Frequency Spectrum
              |
              v
      Digital Filter Selection
      /        |          \
 Low-pass   High-pass   Band-pass
              |
              v
       Filtered Signal
              |
        +-----+------+
        |            |
        v            v
       FFT       Play / Save
```

## DSP Implementation Notes

### Audio
For stereo files, create a mono analysis signal by averaging channels, while preserving the original file separately.

### FFT
For real-valued audio, use a one-sided spectrum:

```python
np.fft.rfft(signal)
np.fft.rfftfreq(n, 1 / fs)
```

### Filters
Use Butterworth filters from `scipy.signal`. Prefer second-order sections and offline zero-phase filtering:

```python
sos = scipy.signal.butter(order, cutoff, btype=..., fs=fs, output="sos")
filtered = scipy.signal.sosfiltfilt(sos, signal)
```

Always validate:

```text
0 < cutoff < Nyquist
```

For band-pass:

```text
0 < low_cutoff < high_cutoff < Nyquist
```

### SNR
When a clean reference exists:

```text
noise = noisy - clean
SNR(dB) = 10 log10(Psignal / Pnoise)
```

For the filtered result:

```text
residual = filtered - clean
SNR_after = 10 log10(Psignal / Presidual)
```

Never invent SNR values for arbitrary real recordings where no clean reference exists.

## GUI
Suggested sections:
- Input
- Signal information
- Original waveform + FFT
- Noise controls
- Filter controls
- Filtered waveform + FFT
- SNR
- Playback/save controls
- Status/error message area

## Definition of Done
The app launches, loads a WAV, plots waveform and FFT, adds noise, applies all three filters, shows the processed results, saves output, handles errors, and has clear README/setup instructions.

## Demo Flow
```text
Load voice.wav
  -> show original waveform
  -> show original FFT
  -> add noise
  -> show noisy waveform + FFT
  -> choose filter + cutoff
  -> apply filter
  -> show filtered waveform + FFT
  -> compare SNR when valid
  -> play original/noisy/filtered
  -> save filtered.wav
```

## Connection to LINA
The project can later be used as a signal-processing pre-processing stage for a speech/voice assistant:

```text
Microphone -> Signal Processing -> Clean Voice -> Speech Recognition -> LINA
```

LINA integration is optional and must not delay completion of the core project.
