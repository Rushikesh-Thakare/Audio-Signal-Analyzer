"""
audio_io.py — Audio Input/Output and Metadata Handling Module

This module provides simple, robust functions for:
1. Loading audio files (WAV format) using SoundFile.
2. Converting multi-channel signals to mono for DSP analysis while preserving data.
3. Calculating basic time-domain statistics like Root Mean Square (RMS).
4. Saving processed audio back to disk safely (preventing clipping distortion).
5. Safe audio playback using sounddevice with graceful failure handling.
"""

import os
from typing import Dict, Any, Tuple
import numpy as np
import soundfile as sf


def calculate_rms(signal: np.ndarray) -> float:
    """
    Calculate the Root Mean Square (RMS) amplitude of a 1D discrete-time signal.

    Formula:
        RMS = sqrt((1 / N) * sum(x[n]^2))

    Args:
        signal: 1D numpy array representing signal amplitudes.

    Returns:
        float: RMS value representing the signal's effective amplitude/energy.
    """
    if signal.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(signal, dtype=np.float64))))


def to_mono(audio_data: np.ndarray) -> np.ndarray:
    """
    Convert audio data to a single mono channel by averaging channels.

    Signals & Systems note:
    For single-channel DSP analysis (FFT, filtering), a 1D discrete-time sequence
    x[n] is required. For stereo audio x[n, c], we compute:
        x_mono[n] = (1 / C) * sum_{c=1}^{C} x[n, c]

    Args:
        audio_data: Numpy array of shape (N,) for mono or (N, C) for multi-channel.

    Returns:
        np.ndarray: 1D float32 numpy array representing mono audio.
    """
    if audio_data.ndim == 1:
        return audio_data.astype(np.float32, copy=True)
    elif audio_data.ndim == 2:
        # Average across channels (axis 1)
        return np.mean(audio_data, axis=1, dtype=np.float32)
    else:
        raise ValueError(
            f"Unsupported audio array dimension: {audio_data.ndim}. "
            "Expected 1D (mono) or 2D (multi-channel) audio array."
        )


def load_audio(filepath: str) -> Dict[str, Any]:
    """
    Load a WAV audio file, validate its structure, and extract essential metadata.

    Args:
        filepath: Path to the audio file on disk.

    Returns:
        dict: Metadata and signal arrays containing:
            - 'data': np.ndarray, original raw audio data (preserving stereo/multi-channel)
            - 'mono_data': np.ndarray, 1D float32 normalized mono analysis signal
            - 'fs': int, sampling rate in Hertz (samples per second)
            - 'samples': int, total number of sample points N
            - 'duration': float, length of audio in seconds (N / fs)
            - 'channels': int, number of audio channels (1 for mono, 2 for stereo)
            - 'rms': float, Root Mean Square amplitude of the mono signal
            - 'filepath': str, validated absolute path to the file

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty, corrupt, or unsupported.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    try:
        # Load audio data as float32 normalized between -1.0 and +1.0
        data, fs = sf.read(filepath, dtype="float32")
    except Exception as exc:
        raise ValueError(f"Failed to read audio file '{filepath}': {exc}") from exc

    if data.size == 0:
        raise ValueError(f"The audio file '{filepath}' is empty (0 samples).")

    # Determine channel count and sample count
    if data.ndim == 1:
        channels = 1
        num_samples = len(data)
    else:
        channels = data.shape[1]
        num_samples = data.shape[0]

    # Convert to mono for standardized 1D DSP processing
    mono_data = to_mono(data)
    duration = float(num_samples) / float(fs)
    rms_val = calculate_rms(mono_data)

    return {
        "data": data,
        "mono_data": mono_data,
        "fs": int(fs),
        "samples": int(num_samples),
        "duration": duration,
        "channels": channels,
        "rms": rms_val,
        "filepath": os.path.abspath(filepath),
    }


def save_audio(filepath: str, audio_data: np.ndarray, fs: int) -> str:
    """
    Save an audio signal array as a standard 16-bit PCM WAV file.

    Ensures that floating point samples are safely clamped to [-1.0, 1.0] to
    prevent digital clipping wrap-around distortion.

    Args:
        filepath: Destination path where the WAV file will be written.
        audio_data: 1D or 2D numpy array containing audio samples.
        fs: Sampling rate in Hz.

    Returns:
        str: Absolute path of the saved WAV file.

    Raises:
        ValueError: If audio_data is empty or invalid.
    """
    if audio_data is None or audio_data.size == 0:
        raise ValueError("Cannot save empty audio data.")

    if fs <= 0:
        raise ValueError(f"Invalid sampling rate: {fs}. Sampling rate must be positive.")

    # Create destination directory if it does not exist
    dest_dir = os.path.dirname(os.path.abspath(filepath))
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)

    # Safe amplitude clamping to avoid audio overflow distortion
    clamped_data = np.clip(audio_data, -1.0, 1.0).astype(np.float32)

    try:
        sf.write(filepath, clamped_data, fs, subtype="PCM_16")
    except Exception as exc:
        raise IOError(f"Failed to save audio to '{filepath}': {exc}") from exc

    return os.path.abspath(filepath)


def play_audio(audio_data: np.ndarray, fs: int) -> bool:
    """
    Safely play audio through the default local output device using sounddevice.

    Does not raise exceptions if no output device is found or if sounddevice
    encounters a hardware error.

    Args:
        audio_data: 1D or 2D numpy array of audio samples.
        fs: Sampling frequency in Hz.

    Returns:
        bool: True if playback began successfully, False if failed.
    """
    try:
        import sounddevice as sd

        # Stop any active playback before starting new playback
        sd.stop()
        sd.play(audio_data, samplerate=fs)
        return True
    except Exception as exc:
        print(f"[Audio Playback Notice] Could not play audio: {exc}")
        return False


def stop_audio() -> None:
    """Stop any actively playing audio stream safely."""
    try:
        import sounddevice as sd

        sd.stop()
    except Exception:
        pass


def is_microphone_available() -> bool:
    """
    Check if a valid audio input device (microphone) is detected on the system.

    Returns:
        bool: True if an audio recording device is available, False otherwise.
    """
    try:
        import sounddevice as sd

        dev = sd.query_devices(kind="input")
        return bool(dev and dev.get("max_input_channels", 0) > 0)
    except Exception:
        return False


def record_audio(duration: float = 3.0, fs: int = 44100) -> np.ndarray:
    """
    Record audio from the local microphone for a specified duration in seconds.

    Args:
        duration: Duration of recording in seconds (default 3.0s).
        fs: Sampling rate in Hz (default 44100 Hz).

    Returns:
        np.ndarray: 1D float32 normalized mono audio array.

    Raises:
        RuntimeError: If no audio input device is available or recording fails.
    """
    if not is_microphone_available():
        raise RuntimeError(
            "No audio recording device (microphone) detected on this system."
        )

    try:
        import sounddevice as sd

        num_samples = int(duration * fs)
        recording = sd.rec(
            num_samples, samplerate=fs, channels=1, dtype="float32", blocking=True
        )
        return recording.flatten()
    except Exception as exc:
        raise RuntimeError(f"Microphone recording failed: {exc}") from exc

