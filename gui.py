"""
gui.py — PySide6 Desktop GUI for Audio Signal Analyzer & Noise Reduction System

Features:
1. Primary Workflow: Load noisy audio (WAV file or 3s live microphone) -> Process & Clean.
2. Speech-Preserving Noise Reduction: STFT spectral gating with soft-knee Wiener masking.
3. Intelligent Noise Profiling: Auto-detection of quietest interval, leading pause, or user-selected range.
4. Classical Butterworth SOS Filtering: Low-pass, High-pass, Band-pass with zero-phase filtering.
5. Dual Objective Evaluation:
   - Real Audio: Unreferenced noise floor attenuation, speech energy retention, and RMS change.
   - Synthetic Benchmarks: Ground-truth SNR before, after, and delta improvement against clean reference.
6. Time-Frequency Visualizations:
   - Overview side-by-side time waveforms and FFT magnitude spectra.
   - Individual stage detail views with dominant peak frequency annotations.
   - Interactive Time-Frequency Spectrograms (STFT power density heatmaps).
   - Bode magnitude response curves with -3 dB cutoff indicators.
7. Safe Audio Playback & Clamped WAV Export:
   - Non-blocking sounddevice playback with device fallback.
   - 16-bit PCM export clamped to [-1.0, 1.0].
"""

import os
import sys
from typing import Optional, Dict, Any

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QIcon, QColor
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QSlider,
    QGroupBox,
    QTabWidget,
    QStatusBar,
    QMessageBox,
    QScrollArea,
    QFrame,
    QSizePolicy,
)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from audio_io import (
    load_audio,
    save_audio,
    play_audio,
    stop_audio,
    record_audio,
    is_microphone_available,
    calculate_rms,
)
from signal_processing import (
    compute_time_axis,
    compute_fft,
    find_peak_frequency,
    add_white_gaussian_noise,
    apply_filter,
    validate_filter_cutoffs,
    calculate_snr,
    evaluate_filtering_performance,
    compute_filter_response,
    detect_noise_segment,
    estimate_noise_profile,
    spectral_gate_denoise,
    denoise_audio,
    calculate_unreferenced_metrics,
    compute_spectrogram,
)


class RecordWorker(QThread):
    """Background worker to record microphone audio without blocking the Qt event loop."""
    recording_finished = Signal(object, object)

    def __init__(self, duration: float = 3.0, fs: int = 44100):
        super().__init__()
        self.duration = duration
        self.fs = fs

    def run(self):
        try:
            arr = record_audio(duration=self.duration, fs=self.fs)
            self.recording_finished.emit(arr, None)
        except Exception as exc:
            self.recording_finished.emit(None, str(exc))


class MplCanvas(FigureCanvas):
    """Matplotlib Figure Canvas configured for clean Signals and Systems plotting."""

    def __init__(self, parent=None, width=8, height=6, dpi=95, rows=1, cols=2):
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self.fig.patch.set_facecolor("#FFFFFF")
        super().__init__(self.fig)
        self.setParent(parent)
        self.axes = self.fig.subplots(rows, cols)

    def clear_canvas(self):
        """Clear all subplots and reset backgrounds."""
        if isinstance(self.axes, np.ndarray):
            for ax in self.axes.flat:
                ax.clear()
        else:
            self.axes.clear()
        self.draw()


class MainWindow(QMainWindow):
    """Main Application Window for the Audio Signal Analyzer & Noise Reduction System."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Signal Analyzer & Noise Reduction System Using FFT")
        self.resize(1340, 880)
        self.setMinimumSize(1080, 720)

        # Application State
        self.audio_meta: Optional[Dict[str, Any]] = None
        self.clean_signal: Optional[np.ndarray] = None  # Primary input audio (raw / noisy)
        self.noisy_signal: Optional[np.ndarray] = None  # Populated if synthetic noise is added
        self.filtered_signal: Optional[np.ndarray] = None  # Cleaned / processed audio output
        self.noise_component: Optional[np.ndarray] = None
        self.benchmark_clean_ref: Optional[np.ndarray] = None  # Ground truth reference for benchmarks
        self.fs: int = 44100
        self.noise_profile_meta: Optional[Dict[str, Any]] = None

        # UI Setup
        self._init_ui()
        self._apply_styling()
        self.show_status("Ready. Please load an audio file (WAV) or record via microphone.", "info")

    def _init_ui(self):
        """Construct main window layout with scrollable sidebar controls and tabbed visualizations."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        # ----------------- Left Panel: Controls (Scrollable) -----------------
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setFixedWidth(400)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(10)

        # Group 1: File Loading & Audio Ingestion
        left_layout.addWidget(self._create_input_group())

        # Group 2: Noise Reduction Engine (Primary Action)
        left_layout.addWidget(self._create_denoising_group())

        # Group 3: Playback & Export Controls
        left_layout.addWidget(self._create_playback_group())

        # Group 4: Quantitative Evaluation & Quality Metrics
        left_layout.addWidget(self._create_snr_group())

        # Group 5: Academic Lab Benchmark Tool (Optional AWGN Injection)
        left_layout.addWidget(self._create_benchmark_group())

        left_layout.addStretch()
        left_scroll.setWidget(left_widget)
        main_layout.addWidget(left_scroll)

        # ----------------- Right Panel: Visualizations -----------------
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        self.tabs = QTabWidget()

        # Tab 0: Comparison (Input vs. Cleaned)
        self.canvas_overview = MplCanvas(self, width=8, height=8, dpi=90, rows=2, cols=2)
        self.tabs.addTab(self.canvas_overview, "Pipeline Comparison (Input vs Cleaned)")

        # Tab 1: Input Audio
        self.canvas_original = MplCanvas(self, width=8, height=6, dpi=95, rows=2, cols=1)
        self.tabs.addTab(self.canvas_original, "Input Audio (Time & FFT)")

        # Tab 2: Cleaned Audio
        self.canvas_filtered = MplCanvas(self, width=8, height=6, dpi=95, rows=2, cols=1)
        self.tabs.addTab(self.canvas_filtered, "Cleaned Audio (Time & FFT)")

        # Tab 3: Spectrogram View (STFT Time-Frequency Heatmap)
        self.canvas_spectrogram = MplCanvas(self, width=8, height=6, dpi=95, rows=2, cols=1)
        self.tabs.addTab(self.canvas_spectrogram, "Spectrogram (Time-Frequency)")

        # Tab 4: Synthetic Noisy View (For AWGN Benchmark Mode)
        self.canvas_noisy = MplCanvas(self, width=8, height=6, dpi=95, rows=2, cols=1)
        self.tabs.addTab(self.canvas_noisy, "Benchmark Noisy Audio")

        # Tab 5: Filter Frequency Response |H(f)|
        self.canvas_response = MplCanvas(self, width=8, height=6, dpi=95, rows=2, cols=1)
        self.tabs.addTab(self.canvas_response, "Filter Response |H(f)|")

        right_layout.addWidget(self.tabs)
        main_layout.addWidget(right_widget, stretch=1)

        # ----------------- Status Bar -----------------
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label, 1)

    # ---------------- UI Group Creators ----------------

    def _create_input_group(self) -> QGroupBox:
        group = QGroupBox("1. Audio Ingestion & Metadata")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        row_top = QHBoxLayout()
        self.btn_load_wav = QPushButton("📂 Load Audio WAV")
        self.btn_load_wav.clicked.connect(self.on_load_wav)
        row_top.addWidget(self.btn_load_wav, stretch=2)

        self.btn_record_mic = QPushButton("🎙️ Record (3s)")
        self.btn_record_mic.setStyleSheet(
            "background-color: #0D9488; color: white; font-weight: 600; font-size: 9pt; border-radius: 6px; padding: 6px 10px;"
        )
        self.btn_record_mic.clicked.connect(self.on_record_mic)
        row_top.addWidget(self.btn_record_mic, stretch=2)

        self.btn_reset = QPushButton("🔄 Reset")
        self.btn_reset.setStyleSheet(
            "background-color: #64748B; color: white; font-weight: 600; font-size: 9pt; border-radius: 6px; padding: 6px 8px;"
        )
        self.btn_reset.clicked.connect(self.on_reset)
        row_top.addWidget(self.btn_reset, stretch=1)
        layout.addLayout(row_top)

        grid = QGridLayout()
        grid.setVerticalSpacing(3)
        grid.setHorizontalSpacing(8)

        self.lbl_file = QLabel("File: None")
        self.lbl_fs = QLabel("Sample Rate: -- Hz")
        self.lbl_duration = QLabel("Duration: -- s")
        self.lbl_samples = QLabel("Samples (N): --")
        self.lbl_channels = QLabel("Channels: --")
        self.lbl_rms = QLabel("RMS Energy: --")
        self.lbl_peak_freq = QLabel("Peak Freq (Input): -- Hz")
        self.lbl_peak_freq_filt = QLabel("Peak Freq (Clean): -- Hz")

        for lbl in (
            self.lbl_file, self.lbl_fs, self.lbl_duration,
            self.lbl_samples, self.lbl_channels, self.lbl_rms,
            self.lbl_peak_freq, self.lbl_peak_freq_filt
        ):
            lbl.setStyleSheet("color: #334155; font-size: 8.5pt;")

        grid.addWidget(self.lbl_file, 0, 0, 1, 2)
        grid.addWidget(self.lbl_fs, 1, 0)
        grid.addWidget(self.lbl_duration, 1, 1)
        grid.addWidget(self.lbl_samples, 2, 0)
        grid.addWidget(self.lbl_channels, 2, 1)
        grid.addWidget(self.lbl_rms, 3, 0)
        grid.addWidget(self.lbl_peak_freq, 3, 1)
        grid.addWidget(self.lbl_peak_freq_filt, 4, 0, 1, 2)

        layout.addLayout(grid)
        return group

    def _create_denoising_group(self) -> QGroupBox:
        group = QGroupBox("2. Noise Reduction & Denoising Engine")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        # Processing Method Selector
        row_method = QHBoxLayout()
        row_method.addWidget(QLabel("Method:"))
        self.combo_method = QComboBox()
        self.combo_method.addItems([
            "Speech Spectral Denoising (STFT Wiener)",
            "Butterworth Filter (SOS IIR)",
        ])
        self.combo_method.currentIndexChanged.connect(self._on_method_changed)
        row_method.addWidget(self.combo_method)
        layout.addLayout(row_method)

        # --- Sub-panel: Spectral Gating Controls ---
        self.panel_spectral = QWidget()
        spectral_layout = QVBoxLayout(self.panel_spectral)
        spectral_layout.setContentsMargins(0, 0, 0, 0)
        spectral_layout.setSpacing(6)

        # Strength Slider
        row_str = QHBoxLayout()
        row_str.addWidget(QLabel("Strength:"))
        self.slider_strength = QSlider(Qt.Horizontal)
        self.slider_strength.setRange(0, 150)
        self.slider_strength.setValue(75)
        self.slider_strength.setSingleStep(5)
        self.lbl_strength_val = QLabel("75% (0.75)")
        self.lbl_strength_val.setFixedWidth(70)
        self.lbl_strength_val.setStyleSheet("font-weight: bold; color: #1E293B;")
        self.slider_strength.valueChanged.connect(self._on_strength_slider_changed)
        row_str.addWidget(self.slider_strength)
        row_str.addWidget(self.lbl_strength_val)
        spectral_layout.addLayout(row_str)

        # Noise Profile Estimation Mode
        row_profile = QHBoxLayout()
        row_profile.addWidget(QLabel("Noise Sample:"))
        self.combo_noise_profile = QComboBox()
        self.combo_noise_profile.addItems([
            "Auto-Detect Quietest Segment",
            "Leading Segment (First 0.3s)",
            "Custom Range [Start – End]",
        ])
        self.combo_noise_profile.currentIndexChanged.connect(self._on_noise_profile_mode_changed)
        row_profile.addWidget(self.combo_noise_profile)
        spectral_layout.addLayout(row_profile)

        # Custom Noise Range SpinBoxes
        self.row_custom_range = QHBoxLayout()
        self.lbl_n_start = QLabel("Start (s):")
        self.spin_noise_start = QDoubleSpinBox()
        self.spin_noise_start.setRange(0.0, 120.0)
        self.spin_noise_start.setValue(0.0)
        self.spin_noise_start.setSingleStep(0.05)
        self.lbl_n_end = QLabel("End (s):")
        self.spin_noise_end = QDoubleSpinBox()
        self.spin_noise_end.setRange(0.01, 120.0)
        self.spin_noise_end.setValue(0.3)
        self.spin_noise_end.setSingleStep(0.05)

        self.row_custom_range.addWidget(self.lbl_n_start)
        self.row_custom_range.addWidget(self.spin_noise_start)
        self.row_custom_range.addWidget(self.lbl_n_end)
        self.row_custom_range.addWidget(self.spin_noise_end)
        spectral_layout.addLayout(self.row_custom_range)
        self._set_custom_range_visible(False)

        self.lbl_noise_sample_hint = QLabel(
            "Noise spectrum is estimated from the quietest pause or user-selected range to prevent attenuating speech."
        )
        self.lbl_noise_sample_hint.setWordWrap(True)
        self.lbl_noise_sample_hint.setStyleSheet("color: #64748B; font-size: 8pt; font-style: italic;")
        spectral_layout.addWidget(self.lbl_noise_sample_hint)

        layout.addWidget(self.panel_spectral)

        # --- Sub-panel: Classical Butterworth Filter Controls ---
        self.panel_butterworth = QWidget()
        butter_layout = QVBoxLayout(self.panel_butterworth)
        butter_layout.setContentsMargins(0, 0, 0, 0)
        butter_layout.setSpacing(6)

        row_filter_type = QHBoxLayout()
        row_filter_type.addWidget(QLabel("Filter Type:"))
        self.combo_filter = QComboBox()
        self.combo_filter.addItems([
            "Low-pass (Butterworth)",
            "High-pass (Butterworth)",
            "Band-pass (Butterworth)",
        ])
        self.combo_filter.currentIndexChanged.connect(self._on_filter_type_changed)
        row_filter_type.addWidget(self.combo_filter)
        butter_layout.addLayout(row_filter_type)

        self.row_cutoff1 = QHBoxLayout()
        self.lbl_cutoff1 = QLabel("Cutoff Freq (Hz):")
        self.spin_cutoff1 = QDoubleSpinBox()
        self.spin_cutoff1.setRange(10.0, 22050.0)
        self.spin_cutoff1.setValue(1000.0)
        self.spin_cutoff1.setSingleStep(50.0)
        self.row_cutoff1.addWidget(self.lbl_cutoff1)
        self.row_cutoff1.addWidget(self.spin_cutoff1)
        butter_layout.addLayout(self.row_cutoff1)

        self.row_cutoff2 = QHBoxLayout()
        self.lbl_cutoff2 = QLabel("High Cutoff (Hz):")
        self.spin_cutoff2 = QDoubleSpinBox()
        self.spin_cutoff2.setRange(20.0, 22050.0)
        self.spin_cutoff2.setValue(3000.0)
        self.spin_cutoff2.setSingleStep(50.0)
        self.row_cutoff2.addWidget(self.lbl_cutoff2)
        self.row_cutoff2.addWidget(self.spin_cutoff2)
        butter_layout.addLayout(self.row_cutoff2)

        self.lbl_nyquist_hint = QLabel("Nyquist Limit: fs / 2 = -- Hz")
        self.lbl_nyquist_hint.setStyleSheet("color: #64748B; font-size: 8pt; font-style: italic;")
        butter_layout.addWidget(self.lbl_nyquist_hint)

        self.spin_cutoff1.valueChanged.connect(self._render_filter_response_canvas)
        self.spin_cutoff2.valueChanged.connect(self._render_filter_response_canvas)

        layout.addWidget(self.panel_butterworth)
        self.panel_butterworth.hide()

        # Primary Process Action Button
        self.btn_process_audio = QPushButton("✨ Process & Clean Audio")
        self.btn_process_audio.setStyleSheet(
            "background-color: #2563EB; color: white; font-weight: bold; font-size: 10pt; "
            "border-radius: 6px; padding: 8px 14px;"
        )
        self.btn_process_audio.setEnabled(False)
        self.btn_process_audio.clicked.connect(self.on_process_audio)
        layout.addWidget(self.btn_process_audio)

        # Alias for backward-compatibility with test suites
        self.btn_apply_filter = self.btn_process_audio

        # Initial control adjustments
        self._on_filter_type_changed(0)
        return group

    def _create_playback_group(self) -> QGroupBox:
        group = QGroupBox("3. Audio Playback & Export")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(4)

        self.btn_play_orig = QPushButton("▶ Play Input Audio")
        self.btn_play_filt = QPushButton("▶ Play Cleaned Audio")
        self.btn_play_noisy = QPushButton("▶ Play Benchmark AWGN")
        self.btn_stop_audio = QPushButton("⏹ Stop Audio")

        self.btn_play_orig.setEnabled(False)
        self.btn_play_filt.setEnabled(False)
        self.btn_play_noisy.setEnabled(False)

        self.btn_play_orig.clicked.connect(lambda: self.on_play_audio("clean"))
        self.btn_play_filt.clicked.connect(lambda: self.on_play_audio("filtered"))
        self.btn_play_noisy.clicked.connect(lambda: self.on_play_audio("noisy"))
        self.btn_stop_audio.clicked.connect(stop_audio)

        grid.addWidget(self.btn_play_orig, 0, 0)
        grid.addWidget(self.btn_play_filt, 0, 1)
        grid.addWidget(self.btn_play_noisy, 1, 0)
        grid.addWidget(self.btn_stop_audio, 1, 1)
        layout.addLayout(grid)

        self.btn_save_wav = QPushButton("💾 Save Cleaned WAV")
        self.btn_save_wav.setStyleSheet(
            "background-color: #059669; color: white; font-weight: 600; font-size: 9pt; border-radius: 6px; padding: 7px 12px;"
        )
        self.btn_save_wav.setEnabled(False)
        self.btn_save_wav.clicked.connect(self.on_save_filtered_wav)
        layout.addWidget(self.btn_save_wav)
        return group

    def _create_snr_group(self) -> QGroupBox:
        group = QGroupBox("4. Quantitative Audio Evaluation")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        self.card_snr = QFrame()
        self.card_snr.setStyleSheet(
            "background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 6px;"
        )
        card_layout = QVBoxLayout(self.card_snr)
        card_layout.setSpacing(3)

        # Real Audio Metrics (Honest measurements)
        self.lbl_noise_attenuation = QLabel("Noise Floor Attenuation: -- dB")
        self.lbl_speech_retention = QLabel("Speech Energy Retention: -- %")
        self.lbl_rms_change = QLabel("Overall RMS Change: -- dB")

        # Ground-Truth / Benchmark Metrics
        self.lbl_snr_before = QLabel("SNR Before: N/A")
        self.lbl_snr_after = QLabel("SNR After: N/A")
        self.lbl_snr_gain = QLabel("SNR Improvement: N/A")

        for lbl in (
            self.lbl_noise_attenuation, self.lbl_speech_retention,
            self.lbl_rms_change, self.lbl_snr_before, self.lbl_snr_after
        ):
            lbl.setStyleSheet("font-size: 8.5pt; font-weight: 600; color: #1E293B;")

        self.lbl_snr_gain.setStyleSheet("font-size: 9.5pt; font-weight: bold; color: #1D4ED8;")

        card_layout.addWidget(self.lbl_noise_attenuation)
        card_layout.addWidget(self.lbl_speech_retention)
        card_layout.addWidget(self.lbl_rms_change)
        card_layout.addWidget(self.lbl_snr_before)
        card_layout.addWidget(self.lbl_snr_after)
        card_layout.addWidget(self.lbl_snr_gain)
        layout.addWidget(self.card_snr)

        self.lbl_snr_note = QLabel(
            "Honesty Notice: Ground-Truth SNR is strictly computed when a clean synthetic reference exists. "
            "For real recordings, physically measured noise floor attenuation is reported."
        )
        self.lbl_snr_note.setWordWrap(True)
        self.lbl_snr_note.setStyleSheet("color: #64748B; font-size: 7.5pt;")
        layout.addWidget(self.lbl_snr_note)
        return group

    def _create_benchmark_group(self) -> QGroupBox:
        group = QGroupBox("5. Academic Lab Benchmark Tool (Optional AWGN)")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        desc = QLabel(
            "Optional: Injects known Additive White Gaussian Noise (AWGN) to objectively verify "
            "delta SNR against ground-truth clean reference."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #64748B; font-size: 8pt;")
        layout.addWidget(desc)

        row = QHBoxLayout()
        row.addWidget(QLabel("Target SNR (dB):"))
        self.spin_snr = QDoubleSpinBox()
        self.spin_snr.setRange(-20.0, 40.0)
        self.spin_snr.setValue(10.0)
        self.spin_snr.setSingleStep(1.0)
        self.spin_snr.setToolTip("Lower dB = more severe noise. Higher dB = cleaner signal.")
        row.addWidget(self.spin_snr)
        layout.addLayout(row)

        self.btn_add_noise = QPushButton("⚡ Inject Known AWGN (Lab Test)")
        self.btn_add_noise.setEnabled(False)
        self.btn_add_noise.clicked.connect(self.on_add_noise)
        layout.addWidget(self.btn_add_noise)
        return group

    def _apply_styling(self):
        """Apply clean modern academic styling with high contrast and readable typography."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #F1F5F9;
            }
            QGroupBox {
                background-color: #FFFFFF;
                border: 1px solid #CBD5E1;
                border-radius: 8px;
                margin-top: 10px;
                padding: 10px 8px 8px 8px;
                font-weight: bold;
                font-size: 9pt;
                color: #0F172A;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QPushButton {
                background-color: #2563EB;
                color: white;
                font-weight: 600;
                font-size: 8.5pt;
                border-radius: 6px;
                padding: 6px 10px;
                border: none;
            }
            QPushButton:hover {
                background-color: #1D4ED8;
            }
            QPushButton:pressed {
                background-color: #1E40AF;
            }
            QPushButton:disabled {
                background-color: #94A3B8;
                color: #F1F5F9;
            }
            QComboBox, QDoubleSpinBox, QSpinBox {
                background-color: #FFFFFF;
                border: 1px solid #CBD5E1;
                border-radius: 4px;
                padding: 4px 6px;
                font-size: 8.5pt;
                color: #0F172A;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #E2E8F0;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #2563EB;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #FFFFFF;
                border: 2px solid #2563EB;
                width: 14px;
                margin-top: -4px;
                margin-bottom: -4px;
                border-radius: 7px;
            }
            QTabWidget::pane {
                border: 1px solid #CBD5E1;
                background-color: #FFFFFF;
                border-radius: 6px;
            }
            QTabBar::tab {
                background-color: #E2E8F0;
                color: #475569;
                font-weight: 600;
                font-size: 8.5pt;
                padding: 8px 14px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }
            QTabBar::tab:selected {
                background-color: #FFFFFF;
                color: #2563EB;
                border: 1px solid #CBD5E1;
                border-bottom: none;
            }
            QStatusBar {
                background-color: #FFFFFF;
                border-top: 1px solid #CBD5E1;
            }
        """)

    # ---------------- Event Handlers ----------------

    def _on_method_changed(self, index: int):
        """Toggle between Spectral Denoising and Classical Butterworth Filter controls."""
        if index == 0:
            self.panel_spectral.show()
            self.panel_butterworth.hide()
        else:
            self.panel_spectral.hide()
            self.panel_butterworth.show()
            self._render_filter_response_canvas()

    def _on_strength_slider_changed(self, value: int):
        """Update strength label when slider moves."""
        strength_flt = value / 100.0
        self.lbl_strength_val.setText(f"{value}% ({strength_flt:.2f})")

    def _on_noise_profile_mode_changed(self, index: int):
        """Show or hide custom noise start/end inputs."""
        is_custom = (index == 2)
        self._set_custom_range_visible(is_custom)

    def _set_custom_range_visible(self, visible: bool):
        """Helper to show/hide custom noise segment inputs."""
        self.lbl_n_start.setVisible(visible)
        self.spin_noise_start.setVisible(visible)
        self.lbl_n_end.setVisible(visible)
        self.spin_noise_end.setVisible(visible)

    def _on_filter_type_changed(self, index: int):
        """Adapt cutoff inputs depending on whether Low/High-pass or Band-pass is selected."""
        filter_text = self.combo_filter.currentText()
        if "Band-pass" in filter_text:
            self.lbl_cutoff1.setText("Low Cutoff (Hz):")
            self.row_cutoff2.setEnabled(True)
            self.lbl_cutoff2.show()
            self.spin_cutoff2.show()
            self.spin_cutoff1.setValue(300.0)
            self.spin_cutoff2.setValue(3000.0)
        else:
            self.lbl_cutoff1.setText("Cutoff Freq (Hz):")
            self.lbl_cutoff2.hide()
            self.spin_cutoff2.hide()
            self.spin_cutoff1.setValue(1000.0)
        self._render_filter_response_canvas()

    def on_load_wav(self):
        """Open file dialog, load audio, update metadata, and render original plots."""
        initial_dir = os.path.abspath("sample_audio")
        if not os.path.isdir(initial_dir):
            initial_dir = os.getcwd()

        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio WAV File",
            initial_dir,
            "WAV Audio Files (*.wav);;All Files (*.*)",
        )

        if not filepath:
            return

        try:
            meta = load_audio(filepath)
            self.audio_meta = meta
            self.clean_signal = meta["mono_data"]
            self.fs = meta["fs"]

            # Reset subsequent stages
            self.noisy_signal = None
            self.filtered_signal = None
            self.noise_component = None
            self.benchmark_clean_ref = None

            # Compute FFT and peak frequency for input signal
            freqs, mags = compute_fft(self.clean_signal, self.fs)
            peak_f, peak_m = find_peak_frequency(freqs, mags)

            # Update Metadata UI
            filename = os.path.basename(filepath)
            self.lbl_file.setText(f"File: {filename}")
            self.lbl_file.setToolTip(filepath)
            self.lbl_fs.setText(f"Sample Rate: {self.fs:,} Hz")
            self.lbl_duration.setText(f"Duration: {meta['duration']:.2f} s")
            self.lbl_samples.setText(f"Samples (N): {meta['samples']:,}")
            ch_str = "Mono (1 ch)" if meta["channels"] == 1 else f"Stereo ({meta['channels']} ch -> Mono downmixed)"
            self.lbl_channels.setText(f"Channels: {ch_str}")
            self.lbl_rms.setText(f"RMS Energy: {meta['rms']:.4f}")
            self.lbl_peak_freq.setText(f"Peak Freq (Input): {peak_f:.1f} Hz")
            self.lbl_peak_freq_filt.setText("Peak Freq (Clean): -- Hz")

            nyquist = self.fs / 2.0
            self.lbl_nyquist_hint.setText(f"Nyquist Limit: fs / 2 = {nyquist:.1f} Hz")
            self.spin_cutoff1.setMaximum(nyquist - 1.0)
            self.spin_cutoff2.setMaximum(nyquist - 1.0)
            self.spin_noise_end.setMaximum(meta["duration"])
            self.spin_noise_end.setValue(min(0.3, meta["duration"]))

            # Reset metrics
            self.lbl_noise_attenuation.setText("Noise Floor Attenuation: -- dB")
            self.lbl_speech_retention.setText("Speech Energy Retention: -- %")
            self.lbl_rms_change.setText("Overall RMS Change: -- dB")
            self.lbl_snr_before.setText("SNR Before: N/A")
            self.lbl_snr_after.setText("SNR After: N/A")
            self.lbl_snr_gain.setText("SNR Improvement: N/A")

            # Enable Action Buttons
            self.btn_process_audio.setEnabled(True)
            self.btn_add_noise.setEnabled(True)
            self.btn_play_orig.setEnabled(True)
            self.btn_play_noisy.setEnabled(False)
            self.btn_play_filt.setEnabled(False)
            self.btn_save_wav.setEnabled(False)

            # Render Plots
            self._render_all_plots()
            self.show_status(
                f"Loaded '{filename}' successfully ({meta['duration']:.2f}s, {self.fs} Hz). Ready to process.",
                "success",
            )

        except Exception as exc:
            self.show_status(f"Error loading file: {exc}", "error")
            QMessageBox.critical(self, "Load Audio Error", f"Could not load audio file:\n\n{exc}")

    def on_record_mic(self):
        """Record live audio through the default microphone for 3 seconds."""
        if not is_microphone_available():
            self.show_status("Microphone not detected. Please use 'Load Audio WAV' instead.", "warning")
            QMessageBox.warning(
                self,
                "Microphone Unavailable",
                "No audio recording input device (microphone) was detected on this system.\n\n"
                "Please connect a microphone or use 'Load Audio WAV' to analyze audio.",
            )
            return

        self.btn_record_mic.setEnabled(False)
        self.btn_load_wav.setEnabled(False)
        self.show_status("🎙️ Recording for 3.0 seconds... Please speak into your microphone now.", "warning")

        self.rec_worker = RecordWorker(duration=3.0, fs=44100)
        self.rec_worker.recording_finished.connect(self._on_record_finished)
        self.rec_worker.start()

    def _on_record_finished(self, audio_data, error_msg):
        """Callback when microphone recording completes."""
        self.btn_record_mic.setEnabled(True)
        self.btn_load_wav.setEnabled(True)

        if error_msg or audio_data is None:
            self.show_status(f"Recording error: {error_msg}", "error")
            QMessageBox.warning(self, "Recording Failed", f"Could not record audio:\n\n{error_msg}")
            return

        self.clean_signal = audio_data
        self.fs = 44100
        self.noisy_signal = None
        self.filtered_signal = None
        self.noise_component = None
        self.benchmark_clean_ref = None

        freqs, mags = compute_fft(self.clean_signal, self.fs)
        peak_f, _ = find_peak_frequency(freqs, mags)
        num_samples = len(self.clean_signal)
        duration = float(num_samples) / float(self.fs)
        rms_val = calculate_rms(self.clean_signal)

        self.audio_meta = {
            "data": self.clean_signal,
            "mono_data": self.clean_signal,
            "fs": self.fs,
            "samples": num_samples,
            "duration": duration,
            "channels": 1,
            "rms": rms_val,
            "filepath": "Microphone_Recording.wav",
        }

        self.lbl_file.setText("File: Microphone_Recording.wav")
        self.lbl_fs.setText(f"Sample Rate: {self.fs:,} Hz")
        self.lbl_duration.setText(f"Duration: {duration:.2f} s")
        self.lbl_samples.setText(f"Samples (N): {num_samples:,}")
        self.lbl_channels.setText("Channels: Mono (Microphone)")
        self.lbl_rms.setText(f"RMS Energy: {rms_val:.4f}")
        self.lbl_peak_freq.setText(f"Peak Freq (Input): {peak_f:.1f} Hz")
        self.lbl_peak_freq_filt.setText("Peak Freq (Clean): -- Hz")

        nyquist = self.fs / 2.0
        self.lbl_nyquist_hint.setText(f"Nyquist Limit: fs / 2 = {nyquist:.1f} Hz")
        self.spin_cutoff1.setMaximum(nyquist - 1.0)
        self.spin_cutoff2.setMaximum(nyquist - 1.0)

        self.lbl_noise_attenuation.setText("Noise Floor Attenuation: -- dB")
        self.lbl_speech_retention.setText("Speech Energy Retention: -- %")
        self.lbl_rms_change.setText("Overall RMS Change: -- dB")
        self.lbl_snr_before.setText("SNR Before: N/A")
        self.lbl_snr_after.setText("SNR After: N/A")
        self.lbl_snr_gain.setText("SNR Improvement: N/A")

        self.btn_process_audio.setEnabled(True)
        self.btn_add_noise.setEnabled(True)
        self.btn_play_orig.setEnabled(True)
        self.btn_play_noisy.setEnabled(False)
        self.btn_play_filt.setEnabled(False)
        self.btn_save_wav.setEnabled(False)

        self._render_all_plots()
        self.show_status(
            f"Microphone audio recorded and loaded successfully ({duration:.1f}s, {self.fs} Hz). Ready to process.",
            "success",
        )

    def on_process_audio(self):
        """
        Primary Action: Process and clean the input audio using selected method.
        Supports both Speech Spectral Gating (default) and Butterworth filtering.
        """
        # Determine target signal (if benchmark AWGN was added, target is noisy_signal)
        target_signal = self.noisy_signal if self.noisy_signal is not None else self.clean_signal
        if target_signal is None:
            self.show_status("Please load or record audio first.", "error")
            return

        method_text = self.combo_method.currentText()
        is_spectral = "Spectral" in method_text

        # Feedback during processing
        self.show_status("Processing audio with noise reduction...", "info")
        QApplication.processEvents()

        try:
            if is_spectral:
                # 1. Speech Spectral Gating
                strength = self.slider_strength.value() / 100.0
                profile_mode_idx = self.combo_noise_profile.currentIndex()

                n_start = None
                n_end = None
                if profile_mode_idx == 1:  # Leading 0.3s
                    n_start = 0.0
                    n_end = 0.3
                elif profile_mode_idx == 2:  # Custom Range
                    n_start = float(self.spin_noise_start.value())
                    n_end = float(self.spin_noise_end.value())

                cleaned, metrics = spectral_gate_denoise(
                    target_signal,
                    fs=self.fs,
                    strength=strength,
                    noise_start_s=n_start,
                    noise_end_s=n_end,
                    spectral_floor=0.05,
                    time_smoothing=True,
                )
                self.filtered_signal = cleaned

                # Update unreferenced quality metrics
                att_db = metrics.get("noise_floor_reduction_db", 0.0)
                ret_ratio = metrics.get("speech_retention_ratio", 1.0) * 100.0
                rms_ch = metrics.get("rms_change_db", 0.0)

                self.lbl_noise_attenuation.setText(f"Noise Floor Attenuation: {att_db:.2f} dB")
                self.lbl_speech_retention.setText(f"Speech Energy Retention: {ret_ratio:.1f}%")
                self.lbl_rms_change.setText(f"Overall RMS Change: {rms_ch:+.2f} dB")

                # If synthetic benchmark clean reference exists, compute ground-truth SNR
                if self.benchmark_clean_ref is not None:
                    perf = evaluate_filtering_performance(
                        self.benchmark_clean_ref, target_signal, self.filtered_signal
                    )
                    self.lbl_snr_before.setText(f"SNR Before: {perf['snr_before_db']:.2f} dB")
                    self.lbl_snr_after.setText(f"SNR After: {perf['snr_after_db']:.2f} dB")
                    gain_sign = "+" if perf["snr_improvement_db"] >= 0 else ""
                    self.lbl_snr_gain.setText(
                        f"SNR Improvement: {gain_sign}{perf['snr_improvement_db']:.2f} dB"
                    )
                else:
                    self.lbl_snr_before.setText("SNR Before: N/A (Real Audio)")
                    self.lbl_snr_after.setText("SNR After: N/A (Real Audio)")
                    self.lbl_snr_gain.setText("SNR Improvement: N/A (Unreferenced)")

                desc = metrics.get("noise_profile", {}).get("method", "Spectral Gating")
                self.show_status(f"Spectral noise reduction complete ({desc}, Strength {strength:.2f}).", "success")

            else:
                # 2. Classical Butterworth SOS Filtering
                filter_text = self.combo_filter.currentText()
                if "Low-pass" in filter_text:
                    f_type = "lowpass"
                    cutoff = float(self.spin_cutoff1.value())
                elif "High-pass" in filter_text:
                    f_type = "highpass"
                    cutoff = float(self.spin_cutoff1.value())
                else:
                    f_type = "bandpass"
                    cutoff = (float(self.spin_cutoff1.value()), float(self.spin_cutoff2.value()))

                norm_type, validated_cutoff = validate_filter_cutoffs(f_type, cutoff, self.fs)
                self.filtered_signal = apply_filter(target_signal, self.fs, norm_type, validated_cutoff, order=4)

                # Compute unreferenced metrics
                unref = calculate_unreferenced_metrics(target_signal, self.filtered_signal, self.fs)
                self.lbl_noise_attenuation.setText(f"Noise Floor Attenuation: {unref['noise_floor_reduction_db']:.2f} dB")
                self.lbl_speech_retention.setText(f"Speech Energy Retention: {unref['speech_retention_ratio'] * 100.0:.1f}%")
                self.lbl_rms_change.setText(f"Overall RMS Change: {unref['rms_change_db']:+.2f} dB")

                if self.benchmark_clean_ref is not None or (self.clean_signal is not None and self.noisy_signal is not None):
                    ref = self.benchmark_clean_ref if self.benchmark_clean_ref is not None else self.clean_signal
                    perf = evaluate_filtering_performance(ref, target_signal, self.filtered_signal)
                    self.lbl_snr_before.setText(f"SNR Before: {perf['snr_before_db']:.2f} dB")
                    self.lbl_snr_after.setText(f"SNR After: {perf['snr_after_db']:.2f} dB")
                    gain_sign = "+" if perf["snr_improvement_db"] >= 0 else ""
                    self.lbl_snr_gain.setText(
                        f"SNR Improvement: {gain_sign}{perf['snr_improvement_db']:.2f} dB"
                    )
                else:
                    self.lbl_snr_before.setText("SNR Before: N/A (Real Audio)")
                    self.lbl_snr_after.setText("SNR After: N/A (Real Audio)")
                    self.lbl_snr_gain.setText("SNR Improvement: N/A (Unreferenced)")

                self.show_status(f"Applied {f_type.capitalize()} Butterworth filter successfully.", "success")

            # Enable playback and export
            self.btn_play_filt.setEnabled(True)
            self.btn_save_wav.setEnabled(True)

            # Update peak frequency for filtered signal
            filt_freqs, filt_mags = compute_fft(self.filtered_signal, self.fs)
            f_peak_filt, _ = find_peak_frequency(filt_freqs, filt_mags)
            self.lbl_peak_freq_filt.setText(f"Peak Freq (Clean): {f_peak_filt:.1f} Hz")

            # Update plots
            self._render_all_plots()

        except ValueError as val_err:
            self.show_status(f"Parameter error: {val_err}", "error")
            QMessageBox.warning(self, "Invalid Parameters", str(val_err))
        except Exception as exc:
            self.show_status(f"Processing failed: {exc}", "error")
            QMessageBox.critical(self, "Processing Error", str(exc))

    def on_apply_filter(self):
        """
        Backward-compatible slot for test suite calls.
        Switches to Butterworth method if combo_filter is active, then executes on_process_audio().
        """
        # If caller selected a filter, ensure mode reflects it
        if self.combo_method.currentIndex() != 1:
            self.combo_method.setCurrentIndex(1)
        self.on_process_audio()

    def on_add_noise(self):
        """
        Academic Lab Benchmark Tool: Injects controlled AWGN into clean reference.
        Enables objective ground-truth SNR before/after evaluation for viva exams.
        """
        if self.clean_signal is None:
            self.show_status("Please load an audio file first.", "error")
            return

        try:
            target_snr = float(self.spin_snr.value())
            # Save original clean signal as ground-truth reference
            self.benchmark_clean_ref = self.clean_signal.astype(np.float32, copy=True)

            noisy, noise = add_white_gaussian_noise(
                self.clean_signal, target_snr_db=target_snr, seed=42
            )
            self.noisy_signal = noisy
            self.noise_component = noise

            # Reset filtered signal
            self.filtered_signal = None
            self.btn_play_noisy.setEnabled(True)
            self.btn_play_filt.setEnabled(False)
            self.btn_save_wav.setEnabled(False)

            # Calculate and display SNR before
            snr_before = calculate_snr(self.benchmark_clean_ref, self.noisy_signal)
            self.lbl_snr_before.setText(f"SNR Before: {snr_before:.2f} dB")
            self.lbl_snr_after.setText("SNR After: Pending Filter")
            self.lbl_snr_gain.setText("SNR Improvement: Pending Filter")

            self._render_all_plots()
            self.show_status(
                f"Added White Gaussian Noise (Target: {target_snr:.1f} dB, Measured: {snr_before:.2f} dB). Ready to clean.",
                "success",
            )

        except Exception as exc:
            self.show_status(f"Noise Error: {exc}", "error")
            QMessageBox.warning(self, "Noise Generation Error", str(exc))

    def on_play_audio(self, stage: str):
        """Safely trigger audio playback using sounddevice with graceful failure handling."""
        if stage == "clean" and self.clean_signal is not None:
            sig = self.clean_signal
            name = "Input Audio"
        elif stage == "noisy" and self.noisy_signal is not None:
            sig = self.noisy_signal
            name = "Benchmark Noisy Audio"
        elif stage == "filtered" and self.filtered_signal is not None:
            sig = self.filtered_signal
            name = "Cleaned Audio"
        else:
            return

        success = play_audio(sig, self.fs)
        if success:
            self.show_status(f"Playing {name}...", "info")
        else:
            self.show_status(f"Audio device unavailable for {name} playback.", "warning")

    def on_save_filtered_wav(self):
        """Save the processed/cleaned audio to a user-chosen WAV destination without overwriting input."""
        if self.filtered_signal is None:
            self.show_status("No processed audio available to save.", "error")
            return

        out_dir = os.path.abspath("output")
        os.makedirs(out_dir, exist_ok=True)
        default_name = os.path.join(out_dir, "cleaned_speech_output.wav")

        dest_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Cleaned Audio As WAV",
            default_name,
            "WAV Audio Files (*.wav);;All Files (*.*)",
        )

        if not dest_path:
            return

        # Safeguard: Never overwrite original input audio file
        if self.audio_meta and "filepath" in self.audio_meta:
            if os.path.abspath(dest_path) == os.path.abspath(self.audio_meta["filepath"]):
                QMessageBox.warning(
                    self,
                    "Overwrite Protection",
                    "Cannot overwrite the original input audio file. Please choose a different filename.",
                )
                return

        try:
            saved_file = save_audio(dest_path, self.filtered_signal, self.fs)
            self.show_status(f"Saved cleaned audio to '{os.path.basename(saved_file)}'.", "success")
            QMessageBox.information(
                self, "Audio Saved", f"Successfully exported cleaned audio:\n\n{saved_file}"
            )
        except Exception as exc:
            self.show_status(f"Failed to save audio: {exc}", "error")
            QMessageBox.critical(self, "Save Error", str(exc))

    def on_reset(self):
        """Reset application state, clear signals, reset metadata, and clear all plots."""
        stop_audio()
        self.audio_meta = None
        self.clean_signal = None
        self.noisy_signal = None
        self.filtered_signal = None
        self.noise_component = None
        self.benchmark_clean_ref = None
        self.fs = 44100

        # Reset labels
        self.lbl_file.setText("File: None")
        self.lbl_fs.setText("Sample Rate: -- Hz")
        self.lbl_duration.setText("Duration: -- s")
        self.lbl_samples.setText("Samples (N): --")
        self.lbl_channels.setText("Channels: --")
        self.lbl_rms.setText("RMS Energy: --")
        self.lbl_peak_freq.setText("Peak Freq (Input): -- Hz")
        self.lbl_peak_freq_filt.setText("Peak Freq (Clean): -- Hz")
        self.lbl_nyquist_hint.setText("Nyquist Limit: fs / 2 = -- Hz")

        self.lbl_noise_attenuation.setText("Noise Floor Attenuation: -- dB")
        self.lbl_speech_retention.setText("Speech Energy Retention: -- %")
        self.lbl_rms_change.setText("Overall RMS Change: -- dB")
        self.lbl_snr_before.setText("SNR Before: N/A")
        self.lbl_snr_after.setText("SNR After: N/A")
        self.lbl_snr_gain.setText("SNR Improvement: N/A")

        # Reset buttons
        self.btn_process_audio.setEnabled(False)
        self.btn_add_noise.setEnabled(False)
        self.btn_play_orig.setEnabled(False)
        self.btn_play_noisy.setEnabled(False)
        self.btn_play_filt.setEnabled(False)
        self.btn_save_wav.setEnabled(False)

        # Clear canvases
        self.canvas_overview.clear_canvas()
        self.canvas_original.clear_canvas()
        self.canvas_noisy.clear_canvas()
        self.canvas_filtered.clear_canvas()
        self.canvas_spectrogram.clear_canvas()
        self._render_filter_response_canvas()

        self.show_status("Reset complete. Workspace cleared.", "info")

    # ---------------- Plot Rendering ----------------

    def _render_all_plots(self):
        """Render plots across comparison overview, single-stage tabs, and spectrogram."""
        if self.clean_signal is None:
            return

        # 1. Comparison Overview Canvas
        self._render_overview_canvas()

        # 2. Input Audio Tab
        self._render_single_stage(
            self.canvas_original, self.clean_signal, "Input Audio (Raw / Noisy)", "#2563EB"
        )

        # 3. Cleaned Audio Tab
        if self.filtered_signal is not None:
            self._render_single_stage(
                self.canvas_filtered, self.filtered_signal, "Cleaned Audio (Processed)", "#059669"
            )
        else:
            self.canvas_filtered.clear_canvas()

        # 4. Benchmark Noisy Canvas (if active)
        if self.noisy_signal is not None:
            self._render_single_stage(
                self.canvas_noisy, self.noisy_signal, "Benchmark Noisy Audio (+AWGN)", "#DC2626"
            )
        else:
            self.canvas_noisy.clear_canvas()

        # 5. Spectrogram View
        self._render_spectrogram_canvas()

        # 6. Filter response (if Butterworth)
        self._render_filter_response_canvas()

    def _render_overview_canvas(self):
        """Render side-by-side comparison on the overview canvas."""
        canvas = self.canvas_overview
        fig = canvas.fig
        fig.clear()

        # Check if benchmark noisy signal is present
        has_bench = self.noisy_signal is not None
        rows = 3 if has_bench else 2
        axes = fig.subplots(rows, 2)

        stages = [
            ("Input Audio", self.clean_signal, "#2563EB"),
        ]
        if has_bench:
            stages.append(("Benchmark AWGN", self.noisy_signal, "#DC2626"))
        stages.append(("Cleaned Audio", self.filtered_signal, "#059669"))

        for row_idx, (title, sig, color) in enumerate(stages):
            ax_time = axes[row_idx, 0]
            ax_freq = axes[row_idx, 1]

            if sig is not None:
                step = max(1, len(sig) // 20000)
                t = compute_time_axis(len(sig), self.fs)[::step]
                sig_plot = sig[::step]

                # Time domain waveform
                ax_time.plot(t, sig_plot, color=color, linewidth=0.8)
                ax_time.set_title(f"{title} — Waveform", fontsize=8.5, fontweight="bold", pad=3)
                ax_time.set_xlabel("Time (s)", fontsize=7.5)
                ax_time.set_ylabel("Amplitude", fontsize=7.5)
                ax_time.grid(True, linestyle="--", alpha=0.5)
                ax_time.tick_params(labelsize=7)

                # Frequency domain FFT
                freqs, mags = compute_fft(sig, self.fs)
                max_display_f = min(self.fs / 2.0, 10000.0)
                mask = freqs <= max_display_f
                f_plot = freqs[mask]
                m_plot = mags[mask]

                ax_freq.plot(f_plot, m_plot, color=color, linewidth=0.8)
                ax_freq.set_title(f"{title} — FFT Magnitude", fontsize=8.5, fontweight="bold", pad=3)
                ax_freq.set_xlabel("Frequency (Hz)", fontsize=7.5)
                ax_freq.set_ylabel("Magnitude", fontsize=7.5)
                ax_freq.grid(True, linestyle="--", alpha=0.5)
                ax_freq.tick_params(labelsize=7)

                peak_f, peak_m = find_peak_frequency(freqs, mags)
                if peak_f > 0 and peak_f <= max_display_f:
                    ax_freq.axvline(peak_f, color="#E11D48", linestyle=":", linewidth=1.0, alpha=0.7)
                    ax_freq.text(
                        0.97, 0.90, f"Peak: {peak_f:.1f} Hz",
                        transform=ax_freq.transAxes,
                        ha="right", va="top",
                        fontsize=7,
                        fontweight="bold",
                        color="#BE123C",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFF1F2", edgecolor="#FECDD3", alpha=0.9),
                    )
            else:
                ax_time.text(
                    0.5, 0.5, f"{title}\n[Pending Processing]",
                    ha="center", va="center", color="#94A3B8", fontsize=9
                )
                ax_time.set_xticks([])
                ax_time.set_yticks([])

                ax_freq.text(
                    0.5, 0.5, f"{title} FFT\n[Pending Processing]",
                    ha="center", va="center", color="#94A3B8", fontsize=9
                )
                ax_freq.set_xticks([])
                ax_freq.set_yticks([])

        fig.tight_layout()
        canvas.draw()

    def _render_single_stage(self, canvas: MplCanvas, sig: np.ndarray, title: str, color: str):
        """Render high-detail 2-row (Time + FFT) plot on individual stage tabs."""
        fig = canvas.fig
        fig.clear()
        axes = fig.subplots(2, 1)
        ax_time, ax_freq = axes[0], axes[1]

        step = max(1, len(sig) // 40000)
        t = compute_time_axis(len(sig), self.fs)[::step]
        ax_time.plot(t, sig[::step], color=color, linewidth=1.0)
        ax_time.set_title(f"{title} — Waveform x[n]", fontsize=10, fontweight="bold")
        ax_time.set_xlabel("Time (s)", fontsize=8.5)
        ax_time.set_ylabel("Amplitude", fontsize=8.5)
        ax_time.grid(True, linestyle="--", alpha=0.6)

        freqs, mags = compute_fft(sig, self.fs)
        max_display_f = min(self.fs / 2.0, 10000.0)
        mask = freqs <= max_display_f

        ax_freq.plot(freqs[mask], mags[mask], color=color, linewidth=1.0)
        ax_freq.set_title(f"{title} — One-Sided FFT Spectrum |X(f)|", fontsize=10, fontweight="bold")
        ax_freq.set_xlabel("Frequency (Hz)", fontsize=8.5)
        ax_freq.set_ylabel("Magnitude", fontsize=8.5)
        ax_freq.grid(True, linestyle="--", alpha=0.6)

        peak_f, peak_m = find_peak_frequency(freqs, mags)
        if peak_f > 0 and peak_f <= max_display_f:
            ax_freq.axvline(peak_f, color="#E11D48", linestyle=":", linewidth=1.2, alpha=0.8)
            ax_freq.text(
                0.97, 0.92, f"Dominant Peak: {peak_f:.1f} Hz (Mag: {peak_m:.3f})",
                transform=ax_freq.transAxes,
                ha="right", va="top",
                fontsize=8.5,
                fontweight="bold",
                color="#BE123C",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF1F2", edgecolor="#FECDD3", alpha=0.9),
            )

        fig.tight_layout()
        canvas.draw()

    def _render_spectrogram_canvas(self):
        """Render Time-Frequency Spectrograms (STFT Power Density) for Input vs Cleaned."""
        canvas = self.canvas_spectrogram
        fig = canvas.fig
        fig.clear()
        axes = fig.subplots(2, 1)
        ax_in, ax_out = axes[0], axes[1]

        # 1. Input Spectrogram
        if self.clean_signal is not None and len(self.clean_signal) > 32:
            f_in, t_in, sxx_in = compute_spectrogram(self.clean_signal, self.fs)
            if len(f_in) > 0 and len(t_in) > 0:
                # Limit to 8000 Hz for voice clarity
                f_mask = f_in <= min(8000.0, self.fs / 2.0)
                im1 = ax_in.pcolormesh(
                    t_in, f_in[f_mask], sxx_in[f_mask, :],
                    shading="gouraud", cmap="viridis"
                )
                ax_in.set_title("Input Audio — Time-Frequency Spectrogram (dB)", fontsize=9.5, fontweight="bold")
                ax_in.set_ylabel("Frequency (Hz)", fontsize=8)
                fig.colorbar(im1, ax=ax_in, label="Power (dB)")

        # 2. Cleaned Spectrogram
        if self.filtered_signal is not None and len(self.filtered_signal) > 32:
            f_out, t_out, sxx_out = compute_spectrogram(self.filtered_signal, self.fs)
            if len(f_out) > 0 and len(t_out) > 0:
                f_mask = f_out <= min(8000.0, self.fs / 2.0)
                im2 = ax_out.pcolormesh(
                    t_out, f_out[f_mask], sxx_out[f_mask, :],
                    shading="gouraud", cmap="viridis"
                )
                ax_out.set_title("Cleaned Audio — Time-Frequency Spectrogram (dB)", fontsize=9.5, fontweight="bold")
                ax_out.set_xlabel("Time (s)", fontsize=8)
                ax_out.set_ylabel("Frequency (Hz)", fontsize=8)
                fig.colorbar(im2, ax=ax_out, label="Power (dB)")
        else:
            ax_out.text(
                0.5, 0.5, "Cleaned Spectrogram [Pending Processing]",
                ha="center", va="center", color="#94A3B8", fontsize=9
            )
            ax_out.set_xticks([])
            ax_out.set_yticks([])

        fig.tight_layout()
        canvas.draw()

    def _render_filter_response_canvas(self):
        """Render Bode magnitude frequency response |H(f)| on Tab 5."""
        if not hasattr(self, "canvas_response"):
            return

        canvas = self.canvas_response
        fig = canvas.fig
        fig.clear()
        ax_mag, ax_eff = fig.subplots(2, 1)

        filter_text = self.combo_filter.currentText()
        if "Low-pass" in filter_text:
            f_type = "lowpass"
            cutoff = float(self.spin_cutoff1.value())
        elif "High-pass" in filter_text:
            f_type = "highpass"
            cutoff = float(self.spin_cutoff1.value())
        else:
            f_type = "bandpass"
            cutoff = (float(self.spin_cutoff1.value()), float(self.spin_cutoff2.value()))

        try:
            freqs, mag_single_db, mag_eff_db = compute_filter_response(
                self.fs, f_type, cutoff, order=4, worN=1024
            )

            ax_mag.plot(freqs, mag_single_db, color="#2563EB", linewidth=1.5, label="Single-Pass |H(f)|")
            ax_mag.axhline(-3.0, color="#E11D48", linestyle="--", linewidth=1.0, label="-3 dB Cutoff")
            ax_mag.set_title(f"Butterworth {f_type.capitalize()} Filter — Single-Pass Magnitude Response", fontsize=9.5, fontweight="bold")
            ax_mag.set_ylabel("Magnitude (dB)", fontsize=8.5)
            ax_mag.set_ylim(-80, 5)
            ax_mag.grid(True, linestyle="--", alpha=0.6)
            ax_mag.legend(loc="lower right", fontsize=8)

            ax_eff.plot(freqs, mag_eff_db, color="#059669", linewidth=1.5, label="Effective Zero-Phase |H_eff(f)| = |H(f)|^2")
            ax_eff.axhline(-6.0, color="#E11D48", linestyle="--", linewidth=1.0, label="-6 dB Effective Point")
            ax_eff.set_title("Effective Zero-Phase Response (sosfiltfilt — Zero Phase Delay)", fontsize=9.5, fontweight="bold")
            ax_eff.set_xlabel("Frequency (Hz)", fontsize=8.5)
            ax_eff.set_ylabel("Magnitude (dB)", fontsize=8.5)
            ax_eff.set_ylim(-100, 5)
            ax_eff.grid(True, linestyle="--", alpha=0.6)
            ax_eff.legend(loc="lower right", fontsize=8)

            fig.tight_layout()
            canvas.draw()
        except Exception:
            pass

    def show_status(self, message: str, level: str = "info"):
        """Display status message in status bar with level color coding."""
        colors = {
            "info": "#1E293B",
            "success": "#15803D",
            "warning": "#B45309",
            "error": "#B91C1C",
        }
        color = colors.get(level, "#1E293B")
        self.status_label.setStyleSheet(f"color: {color}; font-weight: 500; font-size: 8.5pt;")
        self.status_label.setText(f" ● {message}")


def main():
    """Application entry point."""
    app = QApplication.instance() or QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
