"""
tests/test_audio_io.py — Unit Tests for Audio I/O Operations

Tests:
- Loading valid mono and stereo WAV files
- Metadata extraction (fs, samples, duration, channels, RMS)
- Stereo-to-mono downmixing correctness
- Audio saving and clipping protection
- Error handling on non-existent, empty, and invalid audio files
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
import soundfile as sf

from unittest.mock import patch, MagicMock

from audio_io import (
    load_audio,
    save_audio,
    to_mono,
    calculate_rms,
    play_audio,
    stop_audio,
    is_microphone_available,
    record_audio,
)


class TestAudioIO(unittest.TestCase):
    """Test suite for audio loading, saving, and format conversions."""

    def setUp(self):
        """Create a temporary directory and generate test WAV files."""
        self.test_dir = tempfile.mkdtemp()
        self.fs = 8000  # Standard low test sampling rate
        self.duration = 0.5  # 0.5 seconds
        self.num_samples = int(self.fs * self.duration)
        self.t = np.linspace(0, self.duration, self.num_samples, endpoint=False)

        # 1. Create a pure 400 Hz mono sine wave
        self.mono_signal = 0.5 * np.sin(2 * np.pi * 400 * self.t, dtype=np.float32)
        self.mono_wav_path = os.path.join(self.test_dir, "test_mono.wav")
        sf.write(self.mono_wav_path, self.mono_signal, self.fs)

        # 2. Create a stereo WAV file with distinct left and right channels
        left = 0.4 * np.sin(2 * np.pi * 300 * self.t, dtype=np.float32)
        right = 0.4 * np.cos(2 * np.pi * 500 * self.t, dtype=np.float32)
        self.stereo_signal = np.column_stack((left, right))
        self.stereo_wav_path = os.path.join(self.test_dir, "test_stereo.wav")
        sf.write(self.stereo_wav_path, self.stereo_signal, self.fs)

    def tearDown(self):
        """Clean up temporary directory and files."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_load_audio_mono(self):
        """Verify loading mono audio and metadata extraction."""
        meta = load_audio(self.mono_wav_path)
        self.assertEqual(meta["fs"], self.fs)
        self.assertEqual(meta["samples"], self.num_samples)
        self.assertAlmostEqual(meta["duration"], self.duration, places=3)
        self.assertEqual(meta["channels"], 1)
        self.assertEqual(meta["mono_data"].ndim, 1)
        self.assertEqual(len(meta["mono_data"]), self.num_samples)

        # Theoretical RMS for a sine wave of amplitude A: A / sqrt(2)
        expected_rms = 0.5 / np.sqrt(2)
        self.assertAlmostEqual(meta["rms"], expected_rms, places=2)

    def test_load_audio_stereo_to_mono(self):
        """Verify stereo file loading, channel detection, and mono downmixing."""
        meta = load_audio(self.stereo_wav_path)
        self.assertEqual(meta["fs"], self.fs)
        self.assertEqual(meta["channels"], 2)
        self.assertEqual(meta["data"].shape, (self.num_samples, 2))
        self.assertEqual(meta["mono_data"].ndim, 1)
        self.assertEqual(len(meta["mono_data"]), self.num_samples)

        # Test mono averaging calculation
        expected_mono = np.mean(self.stereo_signal, axis=1)
        np.testing.assert_allclose(meta["mono_data"], expected_mono, atol=1e-3)

    def test_to_mono_logic(self):
        """Test the mathematical averaging of stereo channels."""
        stereo_test = np.array([[1.0, -1.0], [0.6, 0.4], [0.0, 0.0]], dtype=np.float32)
        mono_out = to_mono(stereo_test)
        expected = np.array([0.0, 0.5, 0.0], dtype=np.float32)
        np.testing.assert_allclose(mono_out, expected, atol=1e-5)

        # Unsupported 3D array should raise ValueError
        with self.assertRaises(ValueError):
            to_mono(np.zeros((2, 2, 2)))

    def test_calculate_rms(self):
        """Test RMS calculation on known DC and empty signals."""
        dc_signal = np.array([3.0, 3.0, 3.0, 3.0])
        self.assertAlmostEqual(calculate_rms(dc_signal), 3.0)

        empty_signal = np.array([])
        self.assertEqual(calculate_rms(empty_signal), 0.0)

    def test_save_audio_and_clipping(self):
        """Test saving audio file and safe amplitude clamping."""
        out_path = os.path.join(self.test_dir, "output_dir", "saved.wav")
        # Generate signal with out-of-bounds amplitudes (> 1.0 and < -1.0)
        unclamped = np.array([-2.5, -0.5, 0.0, 0.8, 3.2], dtype=np.float32)

        saved_file = save_audio(out_path, unclamped, self.fs)
        self.assertTrue(os.path.exists(saved_file))

        # Read back saved file
        read_data, read_fs = sf.read(saved_file)
        self.assertEqual(read_fs, self.fs)
        self.assertEqual(len(read_data), len(unclamped))
        # Ensure values stay strictly in [-1.0, 1.0]
        self.assertTrue(np.all(read_data <= 1.0 + 1e-4))
        self.assertTrue(np.all(read_data >= -1.0 - 1e-4))

    def test_save_audio_invalid_inputs(self):
        """Test error handling for empty audio or invalid sample rates."""
        out_path = os.path.join(self.test_dir, "bad.wav")
        with self.assertRaises(ValueError):
            save_audio(out_path, np.array([]), self.fs)

        with self.assertRaises(ValueError):
            save_audio(out_path, self.mono_signal, fs=0)

    def test_load_audio_nonexistent_file(self):
        """Ensure loading a non-existent file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_audio(os.path.join(self.test_dir, "does_not_exist.wav"))

    def test_load_audio_invalid_content(self):
        """Ensure loading a corrupt/non-audio file raises ValueError."""
        corrupt_path = os.path.join(self.test_dir, "corrupt.wav")
        with open(corrupt_path, "wb") as f:
            f.write(b"NOT_A_REAL_WAV_HEADER_DATA")

        with self.assertRaises(ValueError):
            load_audio(corrupt_path)

    def test_play_and_stop_audio_safe(self):
        """Ensure play_audio and stop_audio execute without unhandled crashes."""
        # Should return boolean (True or False depending on whether an audio device exists)
        result = play_audio(self.mono_signal, self.fs)
        self.assertIsInstance(result, bool)
        stop_audio()

    def test_is_microphone_available(self):
        """Ensure is_microphone_available returns a boolean without raising errors."""
        res = is_microphone_available()
        self.assertIsInstance(res, bool)

    def test_record_audio_no_mic_error(self):
        """Ensure record_audio raises RuntimeError when no microphone is detected."""
        with patch("audio_io.is_microphone_available", return_value=False):
            with self.assertRaises(RuntimeError) as ctx:
                record_audio(duration=1.0, fs=44100)
            self.assertIn("No audio recording device", str(ctx.exception))

    def test_record_audio_mocked_success(self):
        """Ensure record_audio correctly calls sounddevice and returns flattened 1D array."""
        fake_samples = np.zeros((44100, 1), dtype=np.float32)
        with patch("audio_io.is_microphone_available", return_value=True), \
             patch("sounddevice.rec", return_value=fake_samples) as mock_rec:
            recorded = record_audio(duration=1.0, fs=44100)
            self.assertEqual(recorded.ndim, 1)
            self.assertEqual(len(recorded), 44100)
            mock_rec.assert_called_once_with(44100, samplerate=44100, channels=1, dtype="float32", blocking=True)


if __name__ == "__main__":
    unittest.main()
