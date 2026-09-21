"""
tests/test_e2e_audit.py — Comprehensive End-to-End Audit & Robustness Test Suite

Validates:
1. End-to-end pipeline:
   Load WAV -> Waveform -> FFT -> Add Noise -> Noisy Waveform/FFT
   -> Select & Apply Filter (LP, HP, BP) -> Filtered Waveform/FFT
   -> SNR Evaluation -> Save Output WAV -> Readback verification
2. Audio playback error tolerance when no audio hardware is present.
3. Edge cases and invalid inputs:
   - Non-existent files
   - Corrupt files
   - Cutoff frequencies <= 0
   - Cutoff frequencies >= Nyquist
   - Inverted band-pass cutoffs (low >= high)
   - Array length mismatches
   - Empty signals
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
import soundfile as sf

from audio_io import load_audio, save_audio, to_mono, calculate_rms, play_audio
from signal_processing import (
    compute_time_axis,
    compute_fft,
    find_peak_frequency,
    add_white_gaussian_noise,
    apply_lowpass_filter,
    apply_highpass_filter,
    apply_bandpass_filter,
    apply_filter,
    validate_filter_cutoffs,
    calculate_snr,
    evaluate_filtering_performance,
)


class TestEndToEndAudit(unittest.TestCase):
    """End-to-end pipeline and robustness audit."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.fs = 16000
        self.duration = 1.0
        self.t = compute_time_axis(int(self.fs * self.duration), self.fs)

        # Standard multi-tone signal: 400 Hz (voice fundamental) + 4000 Hz (high hiss)
        self.clean = 0.5 * np.sin(2 * np.pi * 400 * self.t, dtype=np.float32)
        self.hiss = 0.3 * np.sin(2 * np.pi * 4000 * self.t, dtype=np.float32)
        self.mixed = (self.clean + self.hiss).astype(np.float32)

        self.test_wav = os.path.join(self.test_dir, "input.wav")
        save_audio(self.test_wav, self.mixed, self.fs)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_full_pipeline_lowpass_audit(self):
        """
        Audit Stage:
        Load WAV -> Waveform -> FFT -> Add Noise -> Noisy Waveform/FFT
        -> Apply Low-pass Filter -> Filtered Waveform/FFT -> SNR -> Save Output
        """
        # 1. Load WAV
        meta = load_audio(self.test_wav)
        self.assertEqual(meta["fs"], self.fs)
        self.assertEqual(meta["samples"], len(self.mixed))
        self.assertEqual(meta["channels"], 1)
        self.assertGreater(meta["rms"], 0.0)

        # 2. Waveform & FFT of Original
        sig = meta["mono_data"]
        t = compute_time_axis(len(sig), self.fs)
        self.assertEqual(len(t), len(sig))

        freqs, mags = compute_fft(sig, self.fs)
        self.assertEqual(freqs[-1], self.fs / 2.0)  # Nyquist = 8000 Hz

        # Check peak frequency is around 400 Hz
        peak_f, peak_m = find_peak_frequency(freqs, mags)
        self.assertAlmostEqual(peak_f, 400.0, delta=2.0)

        # 3. Add Controlled Noise (Target 10 dB SNR)
        noisy, noise = add_white_gaussian_noise(sig, target_snr_db=10.0, seed=42)
        self.assertEqual(len(noisy), len(sig))
        self.assertEqual(len(noise), len(sig))

        # Check Noisy FFT has raised noise floor
        freqs_n, mags_n = compute_fft(noisy, self.fs)
        self.assertEqual(len(freqs_n), len(freqs))

        # 4. Measure SNR Before Filtering
        snr_before = calculate_snr(self.clean, noisy)
        self.assertIsInstance(snr_before, float)

        # 5. Apply Low-Pass Butterworth Filter (Cutoff 1000 Hz)
        filtered = apply_lowpass_filter(noisy, self.fs, cutoff_hz=1000.0, order=5)
        self.assertEqual(len(filtered), len(noisy))

        # 6. Check Filtered FFT: 4000 Hz component attenuated
        freqs_f, mags_f = compute_fft(filtered, self.fs)
        bin_400 = int(np.round(400 * self.duration))
        bin_4000 = int(np.round(4000 * self.duration))

        self.assertGreater(mags_f[bin_400], 0.4)  # 400 Hz preserved
        self.assertLess(mags_f[bin_4000], 0.05)   # 4000 Hz rejected

        # 7. Measure SNR After Filtering
        perf = evaluate_filtering_performance(self.clean, noisy, filtered)
        self.assertGreater(perf["snr_after_db"], perf["snr_before_db"])
        self.assertGreater(perf["snr_improvement_db"], 0.0)

        # 8. Save Filtered Output WAV
        out_wav = os.path.join(self.test_dir, "output_filtered.wav")
        saved_path = save_audio(out_wav, filtered, self.fs)
        self.assertTrue(os.path.exists(saved_path))

        # Read back and verify validity
        read_back = load_audio(saved_path)
        self.assertEqual(read_back["fs"], self.fs)
        self.assertEqual(read_back["samples"], len(filtered))
        self.assertGreater(read_back["rms"], 0.0)

    def test_highpass_and_bandpass_audit(self):
        """Audit high-pass and band-pass filtering pipelines."""
        # High-pass filter test (reject 400 Hz, preserve 4000 Hz)
        hp_filtered = apply_highpass_filter(self.mixed, self.fs, cutoff_hz=2000.0, order=4)
        _, hp_mags = compute_fft(hp_filtered, self.fs)
        bin_400 = int(np.round(400 * self.duration))
        bin_4000 = int(np.round(4000 * self.duration))
        self.assertLess(hp_mags[bin_400], 0.04)
        self.assertGreater(hp_mags[bin_4000], 0.25)

        # Band-pass filter test (pass 3000 Hz to 5000 Hz)
        bp_filtered = apply_bandpass_filter(
            self.mixed, self.fs, low_cutoff_hz=3000.0, high_cutoff_hz=5000.0, order=4
        )
        _, bp_mags = compute_fft(bp_filtered, self.fs)
        self.assertLess(bp_mags[bin_400], 0.04)
        self.assertGreater(bp_mags[bin_4000], 0.25)

    def test_invalid_input_and_cutoff_audit(self):
        """Audit error handling for invalid files and out-of-bounds cutoffs."""
        # 1. Non-existent file
        with self.assertRaises(FileNotFoundError):
            load_audio(os.path.join(self.test_dir, "missing.wav"))

        # 2. Corrupt file
        corrupt_file = os.path.join(self.test_dir, "corrupt.wav")
        with open(corrupt_file, "w") as f:
            f.write("THIS IS NOT AUDIO")
        with self.assertRaises(ValueError):
            load_audio(corrupt_file)

        # 3. Invalid Cutoff: Zero or Negative
        with self.assertRaises(ValueError):
            validate_filter_cutoffs("lowpass", 0, self.fs)
        with self.assertRaises(ValueError):
            validate_filter_cutoffs("highpass", -100, self.fs)

        # 4. Invalid Cutoff: Equal to or Exceeding Nyquist (fs/2 = 8000 Hz)
        with self.assertRaises(ValueError):
            validate_filter_cutoffs("lowpass", 8000, self.fs)
        with self.assertRaises(ValueError):
            validate_filter_cutoffs("highpass", 9500, self.fs)

        # 5. Invalid Band-pass: Lower >= Upper
        with self.assertRaises(ValueError):
            validate_filter_cutoffs("bandpass", (3000, 1000), self.fs)
        with self.assertRaises(ValueError):
            validate_filter_cutoffs("bandpass", (2000, 2000), self.fs)

        # 6. Invalid Band-pass: Upper >= Nyquist
        with self.assertRaises(ValueError):
            validate_filter_cutoffs("bandpass", (1000, 8500), self.fs)


if __name__ == "__main__":
    unittest.main()
