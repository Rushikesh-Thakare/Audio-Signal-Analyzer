# Implementation Plan

## Phase 0 — Environment
- Inspect OS and Python.
- Create/use `.venv`.
- Install only required packages.
- Confirm imports.

## Phase 1 — Skeleton
Create a simple structure such as:

```text
audio-signal-analyzer/
├── app.py
├── gui.py
├── signal_processing.py
├── audio_io.py
├── requirements.txt
├── README.md
├── AGENTS.md
├── tests/
├── sample_audio/
└── output/
```

## Phase 2 — Audio I/O
Implement:
- WAV load
- WAV save
- metadata
- mono analysis from stereo
- safe playback

Test before continuing.

## Phase 3 — Time Domain
Implement:
- time axis
- waveform
- RMS
- basic statistics

## Phase 4 — FFT
Implement:
- FFT
- frequency axis
- one-sided magnitude spectrum

Verify with a known sine wave.

## Phase 5 — Noise
Implement:
- white Gaussian noise
- configurable strength
- reproducible seed for tests

Keep clean/noisy signals separate.

## Phase 6 — Filters
Implement:
- low-pass Butterworth
- high-pass Butterworth
- band-pass Butterworth
- cutoff validation

## Phase 7 — SNR
Implement SNR only where the clean signal is known.

## Phase 8 — GUI
Add:
- load
- metadata
- waveform
- FFT
- noise controls
- filter controls
- filtered plots
- SNR
- playback
- save

## Phase 9 — Testing
Test:
- speech WAV
- sine wave
- noisy signal
- invalid file
- invalid cutoff
- all filters

## Phase 10 — Documentation
Create:
- README.md
- VIVA.md
- DEMO_SCRIPT.md
- ARCHITECTURE.md

## Time-Saving Rule
If time becomes limited, KEEP:
- WAV upload
- waveform
- FFT
- noise
- filters
- filtered waveform/FFT
- save WAV
- simple GUI

CUT FIRST:
- microphone recording
- spectrogram
- advanced filter-response UI
- LINA integration
