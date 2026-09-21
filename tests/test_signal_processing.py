"""
tests/test_signal_processing.py — Unit Tests for Time Domain and FFT Operations

Tests:
- Time axis generation and spacing
- RMS calculation accuracy against theoretical values
- Time-domain statistics computation
- Fast Fourier Transform (FFT) frequency resolution and magnitude scaling
- Peak frequency detection on known synthetic sine waves (e.g. 440 Hz and 1000 Hz)
- Multi-tone harmonic detection
- Error handling on invalid parameters
"""

import unittest
import numpy as np

from signal_processing import (
    compute_time_axis,
    compute_rms,
    compute_signal_stats,
    compute_fft,
    find_peak_frequency,
    add_white_gaussian_noise,
    create_noisy_signal_pair,
    validate_filter_cutoffs,
    apply_lowpass_filter,
    apply_highpass_filter,
    apply_bandpass_filter,
    apply_filter,
    calculate_signal_power,
    calculate_snr,
    evaluate_filtering_performance,
    compute_filter_response,
)


class TestSignalProcessing(unittest.TestCase):
    """Test suite for time-domain and frequency-domain DSP algorithms."""

    def test_compute_time_axis(self):
        """Verify time axis calculation: length, step size (1/fs), and endpoints."""
        fs = 1000  # 1000 Hz
        num_samples = 500  # 0.5 seconds
        t = compute_time_axis(num_samples, fs)

        self.assertEqual(len(t), num_samples)
        self.assertAlmostEqual(t[0], 0.0)
        # Final point should be (N-1) / fs
        self.assertAlmostEqual(t[-1], 499.0 / 1000.0)
        # Step size should be 1 / fs
        dt = t[1] - t[0]
        self.assertAlmostEqual(dt, 1.0 / fs)

        # Edge cases
        self.assertEqual(len(compute_time_axis(0, fs)), 0)
        with self.assertRaises(ValueError):
            compute_time_axis(-10, fs)
        with self.assertRaises(ValueError):
            compute_time_axis(100, fs=0)

    def test_compute_rms_theoretical(self):
        """Verify RMS calculation matches theoretical values for DC and AC sine waves."""
        fs = 44100
        duration = 1.0
        n = int(fs * duration)
        t = compute_time_axis(n, fs)

        # 1. Pure sine wave: theoretical RMS is Amplitude / sqrt(2)
        amp = 0.8
        sine = amp * np.sin(2 * np.pi * 440 * t)
        expected_rms = amp / np.sqrt(2.0)
        calculated_rms = compute_rms(sine)
        self.assertAlmostEqual(calculated_rms, expected_rms, places=3)

        # 2. DC Signal: theoretical RMS equals the constant amplitude
        dc = np.full(1000, 2.5)
        self.assertAlmostEqual(compute_rms(dc), 2.5)

        # 3. Silence / Zero signal
        zeros = np.zeros(500)
        self.assertEqual(compute_rms(zeros), 0.0)

    def test_compute_signal_stats(self):
        """Verify signal statistics (min, max, peak, mean, rms)."""
        signal = np.array([-1.5, -0.5, 0.0, 0.5, 1.5, 2.0], dtype=np.float64)
        stats = compute_signal_stats(signal)

        self.assertEqual(stats["samples"], 6)
        self.assertAlmostEqual(stats["min"], -1.5)
        self.assertAlmostEqual(stats["max"], 2.0)
        self.assertAlmostEqual(stats["peak"], 2.0)
        self.assertAlmostEqual(stats["mean"], np.mean(signal))
        self.assertAlmostEqual(stats["rms"], compute_rms(signal))

    def test_fft_pure_sine_frequency_and_magnitude(self):
        """
        Verify that FFT accurately locates the peak frequency of a known sine wave
        and normalizes the magnitude to match the input physical amplitude.
        """
        fs = 44100
        duration = 1.0  # 1 second provides 1 Hz bin resolution
        n = int(fs * duration)
        t = compute_time_axis(n, fs)

        # Target frequency: 440 Hz, Amplitude: 0.65
        target_freq = 440.0
        target_amp = 0.65
        sine = target_amp * np.sin(2 * np.pi * target_freq * t)

        freqs, mags = compute_fft(sine, fs)

        # Maximum frequency must be Nyquist = fs / 2
        self.assertAlmostEqual(freqs[-1], fs / 2.0)

        # Find peak frequency
        peak_freq, peak_mag = find_peak_frequency(freqs, mags)

        # Frequency bin resolution = fs / N = 1 Hz
        self.assertAlmostEqual(peak_freq, target_freq, delta=1.0)
        # Peak magnitude should equal input amplitude 0.65
        self.assertAlmostEqual(peak_mag, target_amp, delta=0.02)

    def test_fft_high_frequency_tone(self):
        """Test detection of a 3500 Hz tone at 16 kHz sampling rate."""
        fs = 16000
        duration = 0.5
        n = int(fs * duration)
        t = compute_time_axis(n, fs)

        target_freq = 3500.0
        target_amp = 0.5
        sine = target_amp * np.cos(2 * np.pi * target_freq * t)

        freqs, mags = compute_fft(sine, fs)
        peak_freq, peak_mag = find_peak_frequency(freqs, mags)

        # Frequency resolution is 1 / 0.5 = 2 Hz
        self.assertAlmostEqual(peak_freq, target_freq, delta=2.0)
        self.assertAlmostEqual(peak_mag, target_amp, delta=0.03)

    def test_fft_dual_tone_components(self):
        """Test multi-frequency signal with distinct tones at 300 Hz and 1200 Hz."""
        fs = 8000
        duration = 1.0
        n = int(fs * duration)
        t = compute_time_axis(n, fs)

        f1, a1 = 300.0, 0.4
        f2, a2 = 1200.0, 0.6
        signal = a1 * np.sin(2 * np.pi * f1 * t) + a2 * np.sin(2 * np.pi * f2 * t)

        freqs, mags = compute_fft(signal, fs)

        # Find magnitude around f1 and f2
        bin_f1 = int(np.round(f1 * duration))
        bin_f2 = int(np.round(f2 * duration))

        self.assertAlmostEqual(mags[bin_f1], a1, delta=0.03)
        self.assertAlmostEqual(mags[bin_f2], a2, delta=0.03)

    def test_fft_invalid_inputs(self):
        """Ensure FFT raises ValueError on empty or invalid inputs."""
        with self.assertRaises(ValueError):
            compute_fft(np.array([]), 44100)
        with self.assertRaises(ValueError):
            compute_fft(np.array([1.0, 2.0]), fs=0)

    def test_noise_preserves_clean_signal(self):
        """Verify that adding noise does not mutate the clean signal array."""
        clean = np.array([0.1, 0.5, -0.3, 0.8, -0.2], dtype=np.float32)
        clean_backup = clean.copy()

        noisy, noise = add_white_gaussian_noise(clean, target_snr_db=10.0, seed=123)

        # Clean array must be strictly unchanged
        np.testing.assert_array_equal(clean, clean_backup)
        # Noisy and noise must have same length
        self.assertEqual(len(noisy), len(clean))
        self.assertEqual(len(noise), len(clean))
        # Noisy must equal clean + noise
        np.testing.assert_allclose(noisy, clean + noise, atol=1e-6)

    def test_noise_reproducibility_with_seed(self):
        """Verify that identical random seeds generate identical noise sequences."""
        clean = np.sin(np.linspace(0, 10, 1000, dtype=np.float32))

        noisy_1, noise_1 = add_white_gaussian_noise(clean, target_snr_db=15.0, seed=42)
        noisy_2, noise_2 = add_white_gaussian_noise(clean, target_snr_db=15.0, seed=42)
        noisy_3, noise_3 = add_white_gaussian_noise(clean, target_snr_db=15.0, seed=99)

        np.testing.assert_array_equal(noisy_1, noisy_2)
        np.testing.assert_array_equal(noise_1, noise_2)
        self.assertFalse(np.array_equal(noisy_1, noisy_3))

    def test_noise_power_and_snr_scaling(self):
        """Verify that target SNR closely matches measured empirical SNR for large N."""
        fs = 44100
        duration = 2.0  # 88,200 samples gives high statistical precision for AWGN
        n = int(fs * duration)
        t = compute_time_axis(n, fs)
        clean = 0.7 * np.sin(2 * np.pi * 440 * t)

        target_snr = 15.0  # 15 dB
        noisy, noise = add_white_gaussian_noise(clean, target_snr_db=target_snr, seed=100)

        p_sig = np.mean(clean ** 2)
        p_noise = np.mean(noise ** 2)
        measured_snr = 10.0 * np.log10(p_sig / p_noise)

        # Measured SNR should be within 0.5 dB of target for large N
        self.assertAlmostEqual(measured_snr, target_snr, delta=0.5)

    def test_create_noisy_signal_pair_helper(self):
        """Verify create_noisy_signal_pair returns expected dictionary keys and metrics."""
        clean = 0.5 * np.cos(np.linspace(0, 20, 5000))
        pair = create_noisy_signal_pair(clean, target_snr_db=10.0, seed=7)

        self.assertIn("clean", pair)
        self.assertIn("noisy", pair)
        self.assertIn("noise", pair)
        self.assertIn("target_snr_db", pair)
        self.assertIn("realized_snr_db", pair)
        self.assertEqual(pair["target_snr_db"], 10.0)
        self.assertAlmostEqual(pair["realized_snr_db"], 10.0, delta=0.5)

    def test_noise_empty_signal_error(self):
        """Ensure empty signal raises ValueError."""
        with self.assertRaises(ValueError):
            add_white_gaussian_noise(np.array([]))

    def test_filter_cutoff_validation_rules(self):
        """Verify strict cutoff validation against Nyquist and boundary constraints."""
        fs = 8000  # Nyquist = 4000 Hz

        # Valid cases
        _, c_lp = validate_filter_cutoffs("lowpass", 1000, fs)
        self.assertEqual(c_lp, 1000.0)

        _, c_hp = validate_filter_cutoffs("highpass", 2500, fs)
        self.assertEqual(c_hp, 2500.0)

        _, (low, high) = validate_filter_cutoffs("bandpass", (300, 3000), fs)
        self.assertEqual((low, high), (300.0, 3000.0))

        # Invalid cutoffs <= 0
        with self.assertRaises(ValueError):
            validate_filter_cutoffs("lowpass", 0, fs)
        with self.assertRaises(ValueError):
            validate_filter_cutoffs("highpass", -50, fs)
        with self.assertRaises(ValueError):
            validate_filter_cutoffs("bandpass", (0, 1000), fs)

        # Invalid cutoffs >= Nyquist
        with self.assertRaises(ValueError):
            validate_filter_cutoffs("lowpass", 4000, fs)  # exactly Nyquist
        with self.assertRaises(ValueError):
            validate_filter_cutoffs("highpass", 5000, fs)  # above Nyquist
        with self.assertRaises(ValueError):
            validate_filter_cutoffs("bandpass", (500, 4200), fs)

        # Invalid bandpass ordering (low >= high)
        with self.assertRaises(ValueError):
            validate_filter_cutoffs("bandpass", (2000, 1000), fs)
        with self.assertRaises(ValueError):
            validate_filter_cutoffs("bandpass", (1500, 1500), fs)

        # Unsupported filter type
        with self.assertRaises(ValueError):
            validate_filter_cutoffs("notch_filter", 1000, fs)

    def test_lowpass_filter_attenuation(self):
        """
        Verify low-pass filter preserves low frequencies (200 Hz)
        and sharply attenuates high frequencies (3000 Hz).
        """
        fs = 10000
        duration = 1.0
        n = int(fs * duration)
        t = compute_time_axis(n, fs)

        f_low, a_low = 200.0, 0.6
        f_high, a_high = 3000.0, 0.6
        clean = a_low * np.sin(2 * np.pi * f_low * t) + a_high * np.sin(2 * np.pi * f_high * t)
        clean_backup = clean.copy()

        # Apply lowpass with cutoff at 800 Hz (Nyquist is 5000 Hz)
        filtered = apply_lowpass_filter(clean, fs=fs, cutoff_hz=800.0, order=4)

        # 1. Verify input array was preserved
        np.testing.assert_array_equal(clean, clean_backup)

        # 2. Verify frequency spectrum of filtered signal
        freqs, mags = compute_fft(filtered, fs)
        bin_low = int(np.round(f_low * duration))
        bin_high = int(np.round(f_high * duration))

        # 200 Hz tone should be preserved (> 90% of original amplitude 0.6)
        self.assertGreater(mags[bin_low], 0.54)

        # 3000 Hz tone should be attenuated below 5% of original amplitude (< 0.03)
        self.assertLess(mags[bin_high], 0.03)

    def test_highpass_filter_attenuation(self):
        """
        Verify high-pass filter preserves high frequencies (3000 Hz)
        and sharply attenuates low frequencies (200 Hz).
        """
        fs = 10000
        duration = 1.0
        n = int(fs * duration)
        t = compute_time_axis(n, fs)

        f_low, a_low = 200.0, 0.6
        f_high, a_high = 3000.0, 0.6
        signal = a_low * np.sin(2 * np.pi * f_low * t) + a_high * np.sin(2 * np.pi * f_high * t)

        # Apply highpass with cutoff at 1500 Hz
        filtered = apply_highpass_filter(signal, fs=fs, cutoff_hz=1500.0, order=4)

        freqs, mags = compute_fft(filtered, fs)
        bin_low = int(np.round(f_low * duration))
        bin_high = int(np.round(f_high * duration))

        # 200 Hz tone should be heavily attenuated (< 0.03)
        self.assertLess(mags[bin_low], 0.03)

        # 3000 Hz tone should be preserved (> 0.54)
        self.assertGreater(mags[bin_high], 0.54)

    def test_bandpass_filter_attenuation(self):
        """
        Verify band-pass filter preserves mid-band frequencies (1000 Hz)
        while rejecting both low (100 Hz) and high (4000 Hz) interference.
        """
        fs = 10000
        duration = 1.0
        n = int(fs * duration)
        t = compute_time_axis(n, fs)

        f_low = 100.0
        f_mid = 1000.0
        f_high = 4000.0
        amp = 0.5

        signal = (
            amp * np.sin(2 * np.pi * f_low * t)
            + amp * np.sin(2 * np.pi * f_mid * t)
            + amp * np.sin(2 * np.pi * f_high * t)
        )

        # Passband: 500 Hz to 2000 Hz
        filtered = apply_bandpass_filter(
            signal, fs=fs, low_cutoff_hz=500.0, high_cutoff_hz=2000.0, order=4
        )

        freqs, mags = compute_fft(filtered, fs)
        bin_low = int(np.round(f_low * duration))
        bin_mid = int(np.round(f_mid * duration))
        bin_high = int(np.round(f_high * duration))

        # Low (100 Hz) and High (4000 Hz) attenuated below 0.04
        self.assertLess(mags[bin_low], 0.04)
        self.assertLess(mags[bin_high], 0.04)

        # Mid (1000 Hz) preserved above 0.45
        self.assertGreater(mags[bin_mid], 0.45)

    def test_universal_filter_dispatcher(self):
        """Verify universal apply_filter dispatcher works for all filter types."""
        fs = 8000
        signal = np.sin(2 * np.pi * 300 * compute_time_axis(1000, fs))

        lp = apply_filter(signal, fs, "lowpass", 500)
        self.assertEqual(len(lp), len(signal))

        hp = apply_filter(signal, fs, "highpass", 200)
        self.assertEqual(len(hp), len(signal))

        bp = apply_filter(signal, fs, "bandpass", (200, 500))
        self.assertEqual(len(bp), len(signal))

    def test_snr_identical_signals(self):
        """Verify identical signals yield infinite SNR (zero error power)."""
        clean = np.array([0.2, -0.4, 0.6, -0.8], dtype=np.float32)
        self.assertEqual(calculate_snr(clean, clean), float("inf"))

    def test_snr_zero_power_clean_signal(self):
        """Verify silence/zero clean signal yields -inf SNR."""
        zeros = np.zeros(100, dtype=np.float32)
        noisy = np.ones(100, dtype=np.float32)
        self.assertEqual(calculate_snr(zeros, noisy), float("-inf"))

    def test_snr_dimension_mismatch_raises(self):
        """Ensure mismatched array lengths raise ValueError."""
        with self.assertRaises(ValueError):
            calculate_snr(np.ones(10), np.ones(12))

    def test_snr_empty_signals_raise(self):
        """Ensure empty arrays raise ValueError."""
        with self.assertRaises(ValueError):
            calculate_snr(np.array([]), np.array([]))

    def test_snr_known_noise_measurement(self):
        """
        Verify that adding controlled noise with target SNR 12 dB produces
        an SNR measurement accurate within 0.3 dB.
        """
        fs = 44100
        duration = 2.0
        n = int(fs * duration)
        t = compute_time_axis(n, fs)
        clean = 0.6 * np.sin(2 * np.pi * 500 * t)

        target_snr = 12.0
        noisy, _ = add_white_gaussian_noise(clean, target_snr_db=target_snr, seed=42)

        measured_snr = calculate_snr(clean, noisy)
        self.assertAlmostEqual(measured_snr, target_snr, delta=0.3)

    def test_snr_before_and_after_filtering_improvement(self):
        """
        End-to-End DSP Pipeline Test:
        clean -> add high-frequency noise -> noisy -> low-pass filter -> SNR improvement

        Verifies that:
        1. SNR before filtering is lower than SNR after filtering.
        2. SNR improvement is strictly positive.
        3. No fabricated values are produced.
        """
        fs = 16000
        duration = 1.5
        n = int(fs * duration)
        t = compute_time_axis(n, fs)

        # 1. Clean baseband signal (400 Hz voice fundamental)
        clean = 0.5 * np.sin(2 * np.pi * 400 * t)

        # 2. Heavy out-of-band high-frequency interference (5000 Hz) + noise
        hf_interference = 0.4 * np.sin(2 * np.pi * 5000 * t)
        noisy = (clean + hf_interference).astype(np.float32)

        # 3. Calculate SNR Before
        snr_before = calculate_snr(clean, noisy)

        # 4. Filter with Low-Pass Butterworth (cutoff 1000 Hz)
        filtered = apply_lowpass_filter(signal=noisy, fs=fs, cutoff_hz=1000.0, order=6)

        # 5. Calculate SNR After
        snr_after = calculate_snr(clean, filtered)
        snr_gain = snr_after - snr_before

        # Assert that filtering significantly improved signal quality
        self.assertGreater(snr_after, snr_before)
        self.assertGreater(snr_gain, 10.0)  # Over 10 dB SNR gain expected

        # 6. Test evaluate_filtering_performance helper
        perf = evaluate_filtering_performance(clean, noisy, filtered)
        self.assertAlmostEqual(perf["snr_before_db"], snr_before, places=4)
        self.assertAlmostEqual(perf["snr_after_db"], snr_after, places=4)
        self.assertAlmostEqual(perf["snr_improvement_db"], snr_gain, places=4)
        self.assertGreater(perf["signal_power"], 0.0)
        self.assertGreater(perf["noise_power_before"], perf["residual_power_after"])

    def test_compute_filter_response(self):
        """Verify frequency response calculation and -3 dB cutoff point."""
        fs = 10000
        cutoff = 1000.0
        freqs, mag_single, mag_eff = compute_filter_response(fs, "lowpass", cutoff, order=4, worN=512)

        self.assertEqual(len(freqs), 512)
        self.assertEqual(len(mag_single), 512)
        self.assertEqual(len(mag_eff), 512)

        # DC gain should be approximately 0 dB (gain of 1.0)
        self.assertAlmostEqual(mag_single[0], 0.0, delta=0.1)

        # Find magnitude near cutoff (1000 Hz)
        idx_cutoff = np.argmin(np.abs(freqs - cutoff))
        # Single-pass Butterworth is -3 dB at cutoff
        self.assertAlmostEqual(mag_single[idx_cutoff], -3.01, delta=0.5)
        # Net zero-phase (sosfiltfilt) is -6 dB at cutoff
        self.assertAlmostEqual(mag_eff[idx_cutoff], -6.02, delta=1.0)

        # Stopband attenuation: 4000 Hz should be deeply attenuated (< -40 dB)
        idx_stop = np.argmin(np.abs(freqs - 4000.0))
        self.assertLess(mag_eff[idx_stop], -40.0)


if __name__ == "__main__":
    unittest.main()
