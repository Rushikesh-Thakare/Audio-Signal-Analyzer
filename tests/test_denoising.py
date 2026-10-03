"""
tests/test_denoising.py — Unit Tests for Speech Spectral Gating & Noise Reduction

Verifies:
1. Intelligent noise segment detection (does not blindly assume first 0.3s is noise).
2. User-selected noise interval vs automatic quietest segment fallback.
3. Speech preservation: confirms speech formants are preserved, not just muted.
4. Adjustable reduction strength: higher strength increases attenuation safely.
5. Standalone LINA function (denoise_audio) operates independently without GUI.
6. Honest unreferenced metrics (ground_truth_snr is None, noise floor reduction measured).
7. Robustness against silence, short audio clips, and extreme parameters.
8. Time-frequency spectrogram generation.
"""

import unittest
import numpy as np

from signal_processing import (
    detect_noise_segment,
    estimate_noise_profile,
    spectral_gate_denoise,
    denoise_audio,
    calculate_unreferenced_metrics,
    compute_spectrogram,
    compute_fft,
    find_peak_frequency,
)


class TestSpeechDenoising(unittest.TestCase):
    """Test suite for speech-preserving spectral gating and noise profile estimation."""

    def setUp(self):
        """Set up synthetic speech, noise, and mixed signals for testing."""
        self.fs = 16000  # Standard LINA speech sampling rate
        self.duration = 2.0
        self.n_samples = int(self.fs * self.duration)
        self.t = np.linspace(0, self.duration, self.n_samples, endpoint=False)

        # 1. Synthetic speech with fundamental frequency 200 Hz and harmonics (active 0.5s to 1.8s)
        self.speech = np.zeros(self.n_samples, dtype=np.float32)
        speech_active = (self.t >= 0.5) & (self.t <= 1.8)
        self.speech[speech_active] = (
            0.4 * np.sin(2 * np.pi * 200 * self.t[speech_active])
            + 0.2 * np.sin(2 * np.pi * 400 * self.t[speech_active])
            + 0.1 * np.sin(2 * np.pi * 800 * self.t[speech_active])
        ).astype(np.float32)
        self.speech_mask = speech_active

        # 2. Known stationary background noise (AWGN)
        rng = np.random.default_rng(12345)
        self.noise = rng.normal(0, 0.04, size=self.n_samples).astype(np.float32)

        # 3. Mixed noisy speech signal
        self.noisy_speech = self.speech + self.noise

    def test_detect_noise_segment_leading_pause(self):
        """When initial 0.4s is quiet, detect_noise_segment selects leading segment."""
        start_idx, end_idx, desc = detect_noise_segment(
            self.noisy_speech, self.fs, segment_duration_s=0.3
        )
        self.assertEqual(start_idx, 0)
        self.assertGreater(end_idx, start_idx)
        self.assertIn("Leading", desc)

    def test_detect_noise_segment_immediate_speech(self):
        """
        When user speaks immediately at t=0 and pauses later,
        the algorithm must NOT select t=0. It must detect the quietest segment later.
        """
        # Place speech from 0.0s to 1.2s, silence/noise from 1.2s to 2.0s
        immediate_speech = np.zeros_like(self.t)
        speech_zone = (self.t >= 0.0) & (self.t <= 1.2)
        immediate_speech[speech_zone] = 0.5 * np.sin(2 * np.pi * 300 * self.t[speech_zone])
        noisy_immediate = immediate_speech + self.noise

        start_idx, end_idx, desc = detect_noise_segment(
            noisy_immediate, self.fs, segment_duration_s=0.3
        )
        start_s = start_idx / self.fs
        # Quietest segment must be in the pause region (after 1.0s), not at the start!
        self.assertGreaterEqual(start_s, 1.0)
        self.assertIn("Quietest Segment", desc)

    def test_estimate_noise_profile_user_selection(self):
        """User-specified segment range is respected and validated."""
        profile = estimate_noise_profile(
            self.noisy_speech, self.fs, start_s=0.05, end_s=0.35
        )
        self.assertAlmostEqual(profile["start_s"], 0.05, places=2)
        self.assertAlmostEqual(profile["end_s"], 0.35, places=2)
        self.assertIn("User-Selected", profile["method"])
        self.assertGreater(len(profile["mean_mag"]), 0)
        self.assertGreater(len(profile["std_mag"]), 0)

    def test_speech_preservation_and_noise_reduction(self):
        """
        Core Test: Prove that spectral gating reduces noise while preserving speech.
        A quieter output alone does not prove success; speech harmonics must be retained.
        """
        cleaned, metrics = spectral_gate_denoise(
            self.noisy_speech, self.fs, strength=0.85
        )

        self.assertEqual(len(cleaned), len(self.noisy_speech))

        # Check noise region (0.0s to 0.4s)
        noise_noisy_rms = np.sqrt(np.mean(self.noisy_speech[self.t < 0.4] ** 2))
        noise_clean_rms = np.sqrt(np.mean(cleaned[self.t < 0.4] ** 2))

        # Check speech region
        speech_noisy_rms = np.sqrt(np.mean(self.noisy_speech[self.speech_mask] ** 2))
        speech_clean_rms = np.sqrt(np.mean(cleaned[self.speech_mask] ** 2))

        # 1. Noise must be attenuated significantly (> 5 dB)
        noise_reduction_db = 20 * np.log10(noise_noisy_rms / (noise_clean_rms + 1e-12))
        self.assertGreater(noise_reduction_db, 5.0)

        # 2. Speech must be preserved (speech retention >= 85%)
        speech_retention = speech_clean_rms / speech_noisy_rms
        self.assertGreater(speech_retention, 0.85)

        # 3. Peak frequency of speech must remain 200 Hz
        freqs, mags = compute_fft(cleaned[self.speech_mask], self.fs)
        peak_f, _ = find_peak_frequency(freqs, mags)
        self.assertAlmostEqual(peak_f, 200.0, delta=10.0)

    def test_reduction_strength_variations(self):
        """Test that higher reduction strength attenuates noise more aggressively."""
        cleaned_mild, _ = spectral_gate_denoise(self.noisy_speech, self.fs, strength=0.3)
        cleaned_strong, _ = spectral_gate_denoise(self.noisy_speech, self.fs, strength=1.2)

        noise_mild_rms = np.sqrt(np.mean(cleaned_mild[self.t < 0.4] ** 2))
        noise_strong_rms = np.sqrt(np.mean(cleaned_strong[self.t < 0.4] ** 2))

        self.assertLess(noise_strong_rms, noise_mild_rms)

    def test_zero_strength_bypass(self):
        """Strength = 0.0 must act as a transparent bypass."""
        cleaned_zero, meta = spectral_gate_denoise(self.noisy_speech, self.fs, strength=0.0)
        self.assertEqual(meta["strength"], 0.0)
        np.testing.assert_allclose(cleaned_zero, self.noisy_speech, atol=1e-5)

    def test_standalone_lina_function(self):
        """Test denoise_audio can be called directly without GUI or options."""
        output = denoise_audio(self.noisy_speech, self.fs)
        self.assertIsInstance(output, np.ndarray)
        self.assertEqual(output.dtype, np.float32)
        self.assertEqual(len(output), len(self.noisy_speech))
        # Ensure amplitude stays strictly in [-1.0, 1.0]
        self.assertTrue(np.all(output >= -1.0) and np.all(output <= 1.0))

    def test_unreferenced_metrics_honesty(self):
        """Unreferenced real recordings report Ground-Truth SNR as None (N/A)."""
        metrics = calculate_unreferenced_metrics(
            noisy_signal=self.noisy_speech,
            cleaned_signal=self.noisy_speech * 0.9,
            fs=self.fs,
        )
        self.assertIsNone(metrics["ground_truth_snr"])
        self.assertIn("noise_floor_reduction_db", metrics)
        self.assertIn("speech_retention_ratio", metrics)
        self.assertGreater(metrics["speech_retention_ratio"], 0.0)

    def test_edge_cases(self):
        """Test silence, short clips, and invalid inputs."""
        # 1. Silence
        zeros = np.zeros(1000, dtype=np.float32)
        cleaned_zeros, _ = spectral_gate_denoise(zeros, self.fs)
        self.assertEqual(len(cleaned_zeros), 1000)
        self.assertAlmostEqual(np.max(np.abs(cleaned_zeros)), 0.0, places=4)

        # 2. Short signal (< 64 samples)
        short_sig = np.array([0.1, -0.2, 0.3, -0.1], dtype=np.float32)
        cleaned_short, meta = spectral_gate_denoise(short_sig, self.fs)
        self.assertEqual(len(cleaned_short), 4)

        # 3. Empty signal raises ValueError
        with self.assertRaises(ValueError):
            spectral_gate_denoise(np.array([]), self.fs)

    def test_compute_spectrogram(self):
        """Verify spectrogram computes valid time-frequency grid."""
        freqs, times, sxx_db = compute_spectrogram(self.noisy_speech, self.fs)
        self.assertGreater(len(freqs), 0)
        self.assertGreater(len(times), 0)
        self.assertEqual(sxx_db.shape, (len(freqs), len(times)))
        self.assertTrue(np.all(np.isfinite(sxx_db)))


if __name__ == "__main__":
    unittest.main()
