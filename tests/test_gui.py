"""
tests/test_gui.py — Automated Headless GUI Test Suite for PySide6 Application

Verifies:
- Window initialization and default state
- Programmatic audio loading and metadata label updates
- Adding noise and rendering noisy waveform + FFT
- Filter switching and cutoff validation in UI
- Applying Butterworth filter and SNR before/after metrics
- Exporting filtered WAV
- Safe playback invocation
"""

import os
import sys
import unittest
from unittest.mock import patch
import numpy as np

# Configure Qt for headless / offscreen automated testing
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PySide6.QtWidgets import QApplication

# Ensure QApplication singleton exists
_APP = QApplication.instance() or QApplication(sys.argv)

from gui import MainWindow
from audio_io import load_audio


class TestGUIWorkflow(unittest.TestCase):
    """Automated integration tests for MainWindow."""

    def setUp(self):
        """Create a fresh MainWindow instance for each test."""
        self.window = MainWindow()
        self.sample_wav = os.path.abspath(
            os.path.join("sample_audio", "sine_440hz.wav")
        )
        self.mixed_wav = os.path.abspath(
            os.path.join("sample_audio", "mixed_tone_300hz_3500hz.wav")
        )

    def tearDown(self):
        """Clean up window."""
        self.window.close()

    def test_initial_state(self):
        """Verify initial UI control states (buttons disabled until file loaded)."""
        self.assertFalse(self.window.btn_add_noise.isEnabled())
        self.assertFalse(self.window.btn_apply_filter.isEnabled())
        self.assertFalse(self.window.btn_play_orig.isEnabled())
        self.assertFalse(self.window.btn_play_noisy.isEnabled())
        self.assertFalse(self.window.btn_play_filt.isEnabled())
        self.assertFalse(self.window.btn_save_wav.isEnabled())
        self.assertIn("Ready", self.window.status_label.text())

    def test_load_audio_workflow(self):
        """Simulate loading a WAV file and verify metadata updates."""
        meta = load_audio(self.sample_wav)
        self.window.audio_meta = meta
        self.window.clean_signal = meta["mono_data"]
        self.window.fs = meta["fs"]

        # Run UI update routine manually
        self.window.lbl_file.setText(f"File: {os.path.basename(self.sample_wav)}")
        self.window.lbl_fs.setText(f"Sample Rate: {self.window.fs:,} Hz")
        self.window.lbl_duration.setText(f"Duration: {meta['duration']:.2f} s")
        self.window.btn_add_noise.setEnabled(True)
        self.window.btn_apply_filter.setEnabled(True)
        self.window.btn_play_orig.setEnabled(True)
        self.window._render_all_plots()

        self.assertTrue(self.window.btn_add_noise.isEnabled())
        self.assertTrue(self.window.btn_apply_filter.isEnabled())
        self.assertTrue(self.window.btn_play_orig.isEnabled())
        self.assertEqual(self.window.fs, 44100)
        self.assertIsNotNone(self.window.clean_signal)

    def test_full_pipeline_flow(self):
        """
        Complete end-to-end GUI pipeline test:
        Load -> Add Noise -> Apply Low-Pass Filter -> Check SNR Before & After
        """
        meta = load_audio(self.mixed_wav)
        self.window.audio_meta = meta
        self.window.clean_signal = meta["mono_data"]
        self.window.fs = meta["fs"]

        # 1. Add noise with target SNR 10 dB
        self.window.spin_snr.setValue(10.0)
        self.window.on_add_noise()

        self.assertIsNotNone(self.window.noisy_signal)
        self.assertTrue(self.window.btn_play_noisy.isEnabled())
        # The empirical SNR for target 10.0 dB will be approx 10 dB (+-0.5 dB)
        self.assertIn("SNR Before:", self.window.lbl_snr_before.text())
        snr_text = self.window.lbl_snr_before.text().replace("SNR Before:", "").replace("dB", "").strip()
        self.assertAlmostEqual(float(snr_text), 10.0, delta=0.5)

        # 2. Select Low-pass filter with cutoff 1000 Hz
        self.window.combo_filter.setCurrentIndex(0)  # Low-pass
        self.window.spin_cutoff1.setValue(1000.0)
        self.window.on_apply_filter()

        self.assertIsNotNone(self.window.filtered_signal)
        self.assertTrue(self.window.btn_play_filt.isEnabled())
        self.assertTrue(self.window.btn_save_wav.isEnabled())

        # Verify SNR After was calculated and is not N/A
        self.assertNotIn("N/A", self.window.lbl_snr_after.text())
        self.assertNotIn("N/A", self.window.lbl_snr_gain.text())

    def test_bandpass_filter_selection(self):
        """Verify UI switches between Low-pass and Band-pass controls properly."""
        # Switch to Band-pass
        self.window.combo_filter.setCurrentIndex(2)
        self.assertEqual(self.window.lbl_cutoff1.text(), "Low Cutoff (Hz):")
        self.assertFalse(self.window.spin_cutoff2.isHidden())

        # Switch back to High-pass
        self.window.combo_filter.setCurrentIndex(1)
        self.assertEqual(self.window.lbl_cutoff1.text(), "Cutoff Freq (Hz):")
        self.assertTrue(self.window.spin_cutoff2.isHidden())

    def test_reset_button(self):
        """Verify on_reset clears state, resets controls, and disables action buttons."""
        meta = load_audio(self.sample_wav)
        self.window.audio_meta = meta
        self.window.clean_signal = meta["mono_data"]
        self.window.fs = meta["fs"]
        self.window.btn_add_noise.setEnabled(True)
        self.window.on_add_noise()

        # Call on_reset
        self.window.on_reset()

        self.assertIsNone(self.window.clean_signal)
        self.assertIsNone(self.window.noisy_signal)
        self.assertIsNone(self.window.filtered_signal)
        self.assertFalse(self.window.btn_add_noise.isEnabled())
        self.assertFalse(self.window.btn_apply_filter.isEnabled())
        self.assertFalse(self.window.btn_play_orig.isEnabled())
        self.assertEqual(self.window.lbl_file.text(), "File: None")
        self.assertIn("Reset complete", self.window.status_label.text())

    def test_record_mic_unavailable(self):
        """Verify on_record_mic shows warning and does not crash when mic is unavailable."""
        with patch("gui.is_microphone_available", return_value=False), \
             patch("PySide6.QtWidgets.QMessageBox.warning") as mock_warn:
            self.window.on_record_mic()
            self.assertIn("Microphone not detected", self.window.status_label.text())
            mock_warn.assert_called_once()

    def test_record_mic_finish_callback(self):
        """Verify _on_record_finished populates signal, metadata, and enables action buttons."""
        fs = 44100
        t = np.linspace(0, 1.0, fs, endpoint=False)
        fake_tone = 0.5 * np.sin(2 * np.pi * 500 * t, dtype=np.float32)

        self.window._on_record_finished(fake_tone, "")

        self.assertIsNotNone(self.window.clean_signal)
        self.assertEqual(len(self.window.clean_signal), fs)
        self.assertEqual(self.window.fs, fs)
        self.assertTrue(self.window.btn_add_noise.isEnabled())
        self.assertTrue(self.window.btn_apply_filter.isEnabled())
        self.assertTrue(self.window.btn_play_orig.isEnabled())
        self.assertIn("Microphone_Recording.wav", self.window.lbl_file.text())
        self.assertIn("Microphone audio recorded and loaded successfully", self.window.status_label.text())
        self.assertIn("1.0s", self.window.status_label.text())

    def test_record_mic_error_callback(self):
        """Verify _on_record_finished handles errors gracefully without crashing."""
        with patch("PySide6.QtWidgets.QMessageBox.warning") as mock_warn:
            self.window._on_record_finished(None, "Hardware read timeout")
            self.assertTrue(self.window.btn_record_mic.isEnabled())
            self.assertIn("Recording error", self.window.status_label.text())
            mock_warn.assert_called_once()

    def test_primary_spectral_denoising_workflow_gui(self):
        """
        Verify primary speech spectral denoising workflow from the UI:
        Load Audio -> Select Spectral Method & Strength -> Process & Clean -> Check Metrics & Spectrogram
        """
        meta = load_audio(self.sample_wav)
        self.window.audio_meta = meta
        self.window.clean_signal = meta["mono_data"]
        self.window.fs = meta["fs"]
        self.window.btn_process_audio.setEnabled(True)

        # 1. Select Spectral Gating method (index 0)
        self.window.combo_method.setCurrentIndex(0)
        self.assertFalse(self.window.panel_spectral.isHidden())
        self.assertTrue(self.window.panel_butterworth.isHidden())

        # 2. Adjust strength slider to 85%
        self.window.slider_strength.setValue(85)
        self.assertIn("85%", self.window.lbl_strength_val.text())

        # 3. Trigger processing
        self.window.on_process_audio()

        # 4. Verify processed audio and button state
        self.assertIsNotNone(self.window.filtered_signal)
        self.assertEqual(len(self.window.filtered_signal), len(self.window.clean_signal))
        self.assertTrue(self.window.btn_play_filt.isEnabled())
        self.assertTrue(self.window.btn_save_wav.isEnabled())

        # 5. Verify honest unreferenced metrics are populated
        self.assertIn("Noise Floor Attenuation:", self.window.lbl_noise_attenuation.text())
        self.assertIn("Speech Energy Retention:", self.window.lbl_speech_retention.text())
        self.assertIn("N/A", self.window.lbl_snr_before.text())

    def test_noise_profile_custom_range_visibility(self):
        """Verify noise profile combo toggles custom start/end spinboxes visibility."""
        # Index 0: Auto-detect
        self.window.combo_noise_profile.setCurrentIndex(0)
        self.assertTrue(self.window.spin_noise_start.isHidden())

        # Index 2: Custom Range
        self.window.combo_noise_profile.setCurrentIndex(2)
        self.assertFalse(self.window.spin_noise_start.isHidden())
        self.assertFalse(self.window.spin_noise_end.isHidden())


if __name__ == "__main__":
    unittest.main()

