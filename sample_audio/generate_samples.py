"""
sample_audio/generate_samples.py — Generates test WAV samples for demonstrations and testing.

Creates:
1. sine_440hz.wav: Pure tone at 440 Hz (standard concert pitch A4).
2. synthetic_voice.wav: Multi-harmonic signal simulating voice formants (150 Hz base + formants).
3. mixed_audio.wav: 300 Hz low-frequency tone + 3500 Hz high-frequency interference.
"""

import os
import numpy as np
import soundfile as sf


def generate_all_samples():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    fs = 44100
    duration = 2.5
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # 1. Pure 440 Hz sine wave
    sine_440 = 0.5 * np.sin(2 * np.pi * 440 * t)
    sine_path = os.path.join(output_dir, "sine_440hz.wav")
    sf.write(sine_path, sine_440.astype(np.float32), fs)
    print(f"Generated: {sine_path}")

    # 2. Synthetic Voice / Harmonic Signal (F0=150Hz with decaying harmonics)
    voice = (
        0.4 * np.sin(2 * np.pi * 150 * t)
        + 0.25 * np.sin(2 * np.pi * 300 * t)
        + 0.15 * np.sin(2 * np.pi * 600 * t)
        + 0.10 * np.sin(2 * np.pi * 1200 * t)
    )
    # Apply a gentle envelope to avoid harsh start/stop clicks
    envelope = np.sin(np.pi * t / duration) ** 0.3
    voice = (voice * envelope * 0.8).astype(np.float32)
    voice_path = os.path.join(output_dir, "synthetic_voice.wav")
    sf.write(voice_path, voice, fs)
    print(f"Generated: {voice_path}")

    # 3. Low frequency tone (300 Hz) contaminated by high frequency hum (3500 Hz)
    low_tone = 0.5 * np.sin(2 * np.pi * 300 * t)
    high_hum = 0.25 * np.sin(2 * np.pi * 3500 * t)
    mixed = (low_tone + high_hum).astype(np.float32)
    mixed_path = os.path.join(output_dir, "mixed_tone_300hz_3500hz.wav")
    sf.write(mixed_path, mixed, fs)
    print(f"Generated: {mixed_path}")


if __name__ == "__main__":
    generate_all_samples()
