"""
gui.py — PySide6 Desktop GUI for Audio Signal Analyzer & Noise Reduction System

Features:
1. Load WAV file & validate structure.
2. Display complete metadata: Sampling rate, samples, duration, channels, RMS, peak frequency.
3. Interactive time-domain waveform & frequency-domain FFT plots for Original, Noisy, and Filtered signals.
4. Add controlled Additive White Gaussian Noise (AWGN) with configurable target SNR.
5. Digital Butterworth filter selection (Low-pass, High-pass, Band-pass) with strict cutoff validation.
6. Honest ground-truth SNR before/after evaluation with improvement metrics.
7. Safe audio playback (Play Original, Play Noisy, Play Filtered, Stop) handling unavailable devices.
8. Save processed/filtered WAV file.
9. Informative, color-coded status bar and error messages.
"""

import os
import sys
from typing import Optional, Dict, Any

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QIcon, QColor, QPalette
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

    def __init__(self, parent=None, width=5, height=4, dpi=100, rows=1, cols=2):
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
    """Main Application Window for the Audio Signal Analyzer."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Signal Analyzer and Noise Reduction System Using FFT")
        self.resize(1280, 850)
        self.setMinimumSize(1024, 700)

        # Application State
        self.audio_meta: Optional[Dict[str, Any]] = None
        self.clean_signal: Optional[np.ndarray] = None
        self.noisy_signal: Optional[np.ndarray] = None
        self.filtered_signal: Optional[np.ndarray] = None
        self.noise_component: Optional[np.ndarray] = None
        self.fs: int = 44100

        # UI Setup
        self._init_ui()
        self._apply_styling()
        self.show_status("Ready. Please load a WAV file to begin.", "info")

    def _init_ui(self):
        """Construct main window layout with control sidebar and plot canvas."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        # ----------------- Left Panel: Controls (Scrollable) -----------------
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setFixedWidth(380)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(12)

        # Group 1: File Loading & Metadata
        left_layout.addWidget(self._create_input_group())

        # Group 2: Noise Injection Controls
        left_layout.addWidget(self._create_noise_group())

        # Group 3: Digital Butterworth Filter Controls
        left_layout.addWidget(self._create_filter_group())

        # Group 4: Quantitative SNR Evaluation
        left_layout.addWidget(self._create_snr_group())

        # Group 5: Playback & Export Controls
        left_layout.addWidget(self._create_playback_group())

        left_layout.addStretch()
        left_scroll.setWidget(left_widget)
        main_layout.addWidget(left_scroll)

        # ----------------- Right Panel: Visualizations -----------------
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        self.tabs = QTabWidget()

        # Tab 1: Comprehensive 3-Stage Pipeline (Viva / Professor Presentation View)
        self.canvas_overview = MplCanvas(self, width=8, height=9, dpi=90, rows=3, cols=2)
        self.tabs.addTab(self.canvas_overview, "Pipeline Comparison (Original vs Noisy vs Filtered)")

        # Tab 2: Original Signal Focused View
        self.canvas_original = MplCanvas(self, width=8, height=6, dpi=100, rows=2, cols=1)
        self.tabs.addTab(self.canvas_original, "Stage 1: Original Audio")

        # Tab 3: Noisy Signal Focused View
        self.canvas_noisy = MplCanvas(self, width=8, height=6, dpi=100, rows=2, cols=1)
        self.tabs.addTab(self.canvas_noisy, "Stage 2: Noisy Audio")

        # Tab 4: Filtered Signal Focused View
        self.canvas_filtered = MplCanvas(self, width=8, height=6, dpi=100, rows=2, cols=1)
        self.tabs.addTab(self.canvas_filtered, "Stage 3: Filtered Audio")

        # Tab 5: Filter Frequency Response |H(f)|
        self.canvas_response = MplCanvas(self, width=8, height=6, dpi=100, rows=2, cols=1)
        self.tabs.addTab(self.canvas_response, "Filter Frequency Response |H(f)|")

        right_layout.addWidget(self.tabs)
        main_layout.addWidget(right_widget, stretch=1)

        # ----------------- Status Bar -----------------
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label, 1)

    # ---------------- UI Group Creators ----------------

    def _create_input_group(self) -> QGroupBox:
        group = QGroupBox("1. Audio Input & Signal Metadata")
        layout = QVBoxLayout(group)

        row_top = QHBoxLayout()
        self.btn_load_wav = QPushButton("📂 Load WAV")
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
            "background-color: #64748B; color: white; font-weight: 600; font-size: 9pt; border-radius: 6px; padding: 6px 10px;"
        )
        self.btn_reset.clicked.connect(self.on_reset)
        row_top.addWidget(self.btn_reset, stretch=1)
        layout.addLayout(row_top)

        grid = QGridLayout()
        grid.setVerticalSpacing(4)
        grid.setHorizontalSpacing(8)

        self.lbl_file = QLabel("File: None")
        self.lbl_fs = QLabel("Sample Rate: -- Hz")
        self.lbl_duration = QLabel("Duration: -- s")
        self.lbl_samples = QLabel("Samples (N): --")
        self.lbl_channels = QLabel("Channels: --")
        self.lbl_rms = QLabel("RMS Energy: --")
        self.lbl_peak_freq = QLabel("Peak Freq (Orig): -- Hz")
        self.lbl_peak_freq_filt = QLabel("Peak Freq (Filt): -- Hz")

        # Set text wrapping and compact font for labels
        for lbl in (
            self.lbl_file, self.lbl_fs, self.lbl_duration,
            self.lbl_samples, self.lbl_channels, self.lbl_rms,
            self.lbl_peak_freq, self.lbl_peak_freq_filt
        ):
            lbl.setStyleSheet("color: #333333; font-size: 9pt;")

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

    def _create_noise_group(self) -> QGroupBox:
        group = QGroupBox("2. Controlled Noise (AWGN)")
        layout = QVBoxLayout(group)

        row = QHBoxLayout()
        row.addWidget(QLabel("Target SNR (dB):"))
        self.spin_snr = QDoubleSpinBox()
        self.spin_snr.setRange(-20.0, 40.0)
        self.spin_snr.setValue(10.0)
        self.spin_snr.setSingleStep(1.0)
        self.spin_snr.setToolTip("Lower dB = more severe noise. Higher dB = cleaner signal.")
        row.addWidget(self.spin_snr)
        layout.addLayout(row)

        self.btn_add_noise = QPushButton("⚡ Add White Gaussian Noise")
        self.btn_add_noise.setEnabled(False)
        self.btn_add_noise.clicked.connect(self.on_add_noise)
        layout.addWidget(self.btn_add_noise)
        return group

    def _create_filter_group(self) -> QGroupBox:
        group = QGroupBox("3. Digital Butterworth Filter")
        layout = QVBoxLayout(group)

        row_type = QHBoxLayout()
        row_type.addWidget(QLabel("Filter Type:"))
        self.combo_filter = QComboBox()
        self.combo_filter.addItems([
            "Low-pass (Butterworth)",
            "High-pass (Butterworth)",
            "Band-pass (Butterworth)",
        ])
        self.combo_filter.currentIndexChanged.connect(self._on_filter_type_changed)
        row_type.addWidget(self.combo_filter)
        layout.addLayout(row_type)

        # Cutoff 1 (Primary cutoff or Band-pass Lower cutoff)
        self.row_cutoff1 = QHBoxLayout()
        self.lbl_cutoff1 = QLabel("Cutoff Freq (Hz):")
        self.spin_cutoff1 = QDoubleSpinBox()
        self.spin_cutoff1.setRange(10.0, 22050.0)
        self.spin_cutoff1.setValue(1000.0)
        self.spin_cutoff1.setSingleStep(50.0)
        self.row_cutoff1.addWidget(self.lbl_cutoff1)
        self.row_cutoff1.addWidget(self.spin_cutoff1)
        layout.addLayout(self.row_cutoff1)

        # Cutoff 2 (Band-pass Upper cutoff)
        self.row_cutoff2 = QHBoxLayout()
        self.lbl_cutoff2 = QLabel("High Cutoff (Hz):")
        self.spin_cutoff2 = QDoubleSpinBox()
        self.spin_cutoff2.setRange(20.0, 22050.0)
        self.spin_cutoff2.setValue(3000.0)
        self.spin_cutoff2.setSingleStep(50.0)
        self.row_cutoff2.addWidget(self.lbl_cutoff2)
        self.row_cutoff2.addWidget(self.spin_cutoff2)
        layout.addLayout(self.row_cutoff2)

        # Nyquist hint
        self.lbl_nyquist_hint = QLabel("Nyquist Limit: fs / 2 = -- Hz")
        self.lbl_nyquist_hint.setStyleSheet("color: #666666; font-size: 8pt; font-style: italic;")
        layout.addWidget(self.lbl_nyquist_hint)

        self.spin_cutoff1.valueChanged.connect(self._render_filter_response_canvas)
        self.spin_cutoff2.valueChanged.connect(self._render_filter_response_canvas)

        self.btn_apply_filter = QPushButton("🔧 Apply Digital Filter")
        self.btn_apply_filter.setEnabled(False)
        self.btn_apply_filter.clicked.connect(self.on_apply_filter)
        layout.addWidget(self.btn_apply_filter)

        # Initial control state (Low-pass default hides High Cutoff)
        self._on_filter_type_changed(0)
        self._render_filter_response_canvas()
        return group

    def _create_snr_group(self) -> QGroupBox:
        group = QGroupBox("4. Quantitative SNR Evaluation")
        layout = QVBoxLayout(group)

        self.card_snr = QFrame()
        self.card_snr.setStyleSheet(
            "background-color: #F8F9FA; border: 1px solid #E2E8F0; border-radius: 6px; padding: 6px;"
        )
        card_layout = QVBoxLayout(self.card_snr)
        card_layout.setSpacing(4)

        self.lbl_snr_before = QLabel("SNR Before: N/A")
        self.lbl_snr_after = QLabel("SNR After: N/A")
        self.lbl_snr_gain = QLabel("SNR Improvement: N/A")

        for lbl in (self.lbl_snr_before, self.lbl_snr_after):
            lbl.setStyleSheet("font-size: 9pt; font-weight: bold; color: #2D3748;")

        self.lbl_snr_gain.setStyleSheet("font-size: 10pt; font-weight: bold; color: #2B6CB0;")

        card_layout.addWidget(self.lbl_snr_before)
        card_layout.addWidget(self.lbl_snr_after)
        card_layout.addWidget(self.lbl_snr_gain)
        layout.addWidget(self.card_snr)

        self.lbl_snr_note = QLabel(
            "Note: SNR is calculated strictly against known clean reference (no fabricated values)."
        )
        self.lbl_snr_note.setWordWrap(True)
        self.lbl_snr_note.setStyleSheet("color: #718096; font-size: 8pt;")
        layout.addWidget(self.lbl_snr_note)
        return group

    def _create_playback_group(self) -> QGroupBox:
        group = QGroupBox("5. Playback & Export")
        layout = QVBoxLayout(group)

        grid = QGridLayout()
        self.btn_play_orig = QPushButton("▶ Play Original")
        self.btn_play_noisy = QPushButton("▶ Play Noisy")
        self.btn_play_filt = QPushButton("▶ Play Filtered")
        self.btn_stop_audio = QPushButton("⏹ Stop Audio")

        self.btn_play_orig.setEnabled(False)
        self.btn_play_noisy.setEnabled(False)
        self.btn_play_filt.setEnabled(False)

        self.btn_play_orig.clicked.connect(lambda: self.on_play_audio("clean"))
        self.btn_play_noisy.clicked.connect(lambda: self.on_play_audio("noisy"))
        self.btn_play_filt.clicked.connect(lambda: self.on_play_audio("filtered"))
        self.btn_stop_audio.clicked.connect(stop_audio)

        grid.addWidget(self.btn_play_orig, 0, 0)
        grid.addWidget(self.btn_play_noisy, 0, 1)
        grid.addWidget(self.btn_play_filt, 1, 0)
        grid.addWidget(self.btn_stop_audio, 1, 1)
        layout.addLayout(grid)

        self.btn_save_wav = QPushButton("💾 Save Filtered WAV")
        self.btn_save_wav.setEnabled(False)
        self.btn_save_wav.clicked.connect(self.on_save_filtered_wav)
        layout.addWidget(self.btn_save_wav)
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
                margin-top: 12px;
                padding: 10px 8px 8px 8px;
                font-weight: bold;
                font-size: 9pt;
                color: #1E293B;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px;
            }
            QPushButton {
                background-color: #2563EB;
                color: white;
                font-weight: 600;
                font-size: 9pt;
                border-radius: 6px;
                padding: 6px 12px;
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
                color: #E2E8F0;
            }
            QComboBox, QDoubleSpinBox, QSpinBox {
                background-color: #FFFFFF;
                border: 1px solid #CBD5E1;
                border-radius: 4px;
                padding: 4px 6px;
                font-size: 9pt;
                color: #0F172A;
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
                font-size: 9pt;
                padding: 8px 16px;
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

    def _on_filter_type_changed(self, index: int):
        """Adapt cutoff inputs depending on whether a single-cutoff or band-pass filter is selected."""
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
            if "High-pass" in filter_text:
                self.spin_cutoff1.setValue(1000.0)
            else:
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

            # Compute FFT and peak frequency for original signal
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
            self.lbl_peak_freq.setText(f"Peak Freq (Orig): {peak_f:.1f} Hz")
            self.lbl_peak_freq_filt.setText("Peak Freq (Filt): -- Hz")

            nyquist = self.fs / 2.0
            self.lbl_nyquist_hint.setText(f"Nyquist Limit: fs / 2 = {nyquist:.1f} Hz")
            self.spin_cutoff1.setMaximum(nyquist - 1.0)
            self.spin_cutoff2.setMaximum(nyquist - 1.0)

            # Reset SNR and Buttons
            self.lbl_snr_before.setText("SNR Before: N/A")
            self.lbl_snr_after.setText("SNR After: N/A")
            self.lbl_snr_gain.setText("SNR Improvement: N/A")

            self.btn_add_noise.setEnabled(True)
            self.btn_apply_filter.setEnabled(True)
            self.btn_play_orig.setEnabled(True)
            self.btn_play_noisy.setEnabled(False)
            self.btn_play_filt.setEnabled(False)
            self.btn_save_wav.setEnabled(False)

            # Render Plots
            self._render_all_plots()
            self.show_status(f"Loaded '{filename}' successfully ({meta['duration']:.2f}s, {self.fs} Hz).", "success")

        except Exception as exc:
            self.show_status(f"Error loading file: {exc}", "error")
            QMessageBox.critical(self, "Load Audio Error", f"Could not load audio file:\n\n{exc}")

    def on_record_mic(self):
        """Record live audio through the default microphone for 3 seconds."""
        if not is_microphone_available():
            self.show_status("Microphone not detected. Please use 'Load WAV' instead.", "warning")
            QMessageBox.warning(
                self,
                "Microphone Unavailable",
                "No audio recording input device (microphone) was detected on this system.\n\n"
                "Please connect a microphone or use 'Load WAV' to analyze audio.",
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
        self.lbl_peak_freq.setText(f"Peak Freq (Orig): {peak_f:.1f} Hz")
        self.lbl_peak_freq_filt.setText("Peak Freq (Filt): -- Hz")

        nyquist = self.fs / 2.0
        self.lbl_nyquist_hint.setText(f"Nyquist Limit: fs / 2 = {nyquist:.1f} Hz")
        self.spin_cutoff1.setMaximum(nyquist - 1.0)
        self.spin_cutoff2.setMaximum(nyquist - 1.0)

        self.lbl_snr_before.setText("SNR Before: N/A")
        self.lbl_snr_after.setText("SNR After: N/A")
        self.lbl_snr_gain.setText("SNR Improvement: N/A")

        self.btn_add_noise.setEnabled(True)
        self.btn_apply_filter.setEnabled(True)
        self.btn_play_orig.setEnabled(True)
        self.btn_play_noisy.setEnabled(False)
        self.btn_play_filt.setEnabled(False)
        self.btn_save_wav.setEnabled(False)

        self._render_all_plots()
        self.show_status(f"Microphone audio recorded and loaded successfully ({duration:.1f}s, {self.fs} Hz).", "success")

    def on_add_noise(self):
        """Add Additive White Gaussian Noise to clean signal with configured target SNR."""
        if self.clean_signal is None:
            self.show_status("Please load an audio file first.", "error")
            return

        try:
            target_snr = float(self.spin_snr.value())
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

            # Update SNR before
            snr_before = calculate_snr(self.clean_signal, self.noisy_signal)
            self.lbl_snr_before.setText(f"SNR Before: {snr_before:.2f} dB")
            self.lbl_snr_after.setText("SNR After: Pending Filter")
            self.lbl_snr_gain.setText("SNR Improvement: Pending Filter")

            self._render_all_plots()
            self.show_status(
                f"Added White Gaussian Noise (Target: {target_snr:.1f} dB, Measured: {snr_before:.2f} dB).",
                "success"
            )

        except Exception as exc:
            self.show_status(f"Noise Error: {exc}", "error")
            QMessageBox.warning(self, "Noise Generation Error", str(exc))

    def on_apply_filter(self):
        """Validate cutoffs and apply Butterworth filter to contaminated (or clean) signal."""
        # Filter noisy signal if noise was added; otherwise filter original signal
        target_signal = self.noisy_signal if self.noisy_signal is not None else self.clean_signal
        if target_signal is None:
            self.show_status("Please load an audio file first.", "error")
            return

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
            # 1. Strict Cutoff Validation
            norm_type, validated_cutoff = validate_filter_cutoffs(f_type, cutoff, self.fs)

            # 2. Apply Zero-Phase Butterworth Filter
            filtered = apply_filter(target_signal, self.fs, norm_type, validated_cutoff, order=4)
            self.filtered_signal = filtered

            self.btn_play_filt.setEnabled(True)
            self.btn_save_wav.setEnabled(True)

            # 3. SNR Evaluation (strictly if clean reference exists and noise was added)
            if self.clean_signal is not None and self.noisy_signal is not None:
                perf = evaluate_filtering_performance(
                    self.clean_signal, self.noisy_signal, self.filtered_signal
                )
                self.lbl_snr_before.setText(f"SNR Before: {perf['snr_before_db']:.2f} dB")
                self.lbl_snr_after.setText(f"SNR After: {perf['snr_after_db']:.2f} dB")
                gain_sign = "+" if perf["snr_improvement_db"] >= 0 else ""
                self.lbl_snr_gain.setText(
                    f"SNR Improvement: {gain_sign}{perf['snr_improvement_db']:.2f} dB"
                )
            else:
                self.lbl_snr_before.setText("SNR Before: N/A")
                self.lbl_snr_after.setText("SNR After: N/A")
                self.lbl_snr_gain.setText("SNR Improvement: N/A (Ground-truth noise not added)")

            # Compute and update peak frequency for filtered signal
            filt_freqs, filt_mags = compute_fft(self.filtered_signal, self.fs)
            f_peak_filt, _ = find_peak_frequency(filt_freqs, filt_mags)
            self.lbl_peak_freq_filt.setText(f"Peak Freq (Filt): {f_peak_filt:.1f} Hz")

            self._render_all_plots()
            self.show_status(f"Applied {f_type.capitalize()} filter successfully.", "success")

        except ValueError as val_err:
            self.show_status(f"Filter parameter error: {val_err}", "error")
            QMessageBox.warning(self, "Invalid Filter Cutoff", str(val_err))
        except Exception as exc:
            self.show_status(f"Filtering failed: {exc}", "error")
            QMessageBox.critical(self, "Filter Execution Error", str(exc))

    def on_play_audio(self, stage: str):
        """Safely trigger audio playback using sounddevice with graceful failure handling."""
        if stage == "clean" and self.clean_signal is not None:
            sig = self.clean_signal
            name = "Original Audio"
        elif stage == "noisy" and self.noisy_signal is not None:
            sig = self.noisy_signal
            name = "Noisy Audio"
        elif stage == "filtered" and self.filtered_signal is not None:
            sig = self.filtered_signal
            name = "Filtered Audio"
        else:
            return

        success = play_audio(sig, self.fs)
        if success:
            self.show_status(f"Playing {name}...", "info")
        else:
            self.show_status(f"Audio device unavailable for {name} playback.", "warning")

    def on_save_filtered_wav(self):
        """Save the processed filtered audio to a user-chosen WAV destination."""
        if self.filtered_signal is None:
            self.show_status("No filtered audio available to save.", "error")
            return

        out_dir = os.path.abspath("output")
        os.makedirs(out_dir, exist_ok=True)
        default_name = os.path.join(out_dir, "filtered_output.wav")

        dest_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Filtered Audio As WAV",
            default_name,
            "WAV Audio Files (*.wav);;All Files (*.*)",
        )

        if not dest_path:
            return

        try:
            saved_file = save_audio(dest_path, self.filtered_signal, self.fs)
            self.show_status(f"Saved filtered audio to '{os.path.basename(saved_file)}'.", "success")
            QMessageBox.information(
                self, "Audio Saved", f"Successfully saved filtered audio:\n\n{saved_file}"
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
        self.fs = 44100

        # Reset metadata labels
        self.lbl_file.setText("File: None")
        self.lbl_fs.setText("Sample Rate: -- Hz")
        self.lbl_duration.setText("Duration: -- s")
        self.lbl_samples.setText("Samples (N): --")
        self.lbl_channels.setText("Channels: --")
        self.lbl_rms.setText("RMS Energy: --")
        self.lbl_peak_freq.setText("Peak Freq (Orig): -- Hz")
        self.lbl_peak_freq_filt.setText("Peak Freq (Filt): -- Hz")
        self.lbl_nyquist_hint.setText("Nyquist Limit: fs / 2 = -- Hz")

        # Reset SNR labels
        self.lbl_snr_before.setText("SNR Before: N/A")
        self.lbl_snr_after.setText("SNR After: N/A")
        self.lbl_snr_gain.setText("SNR Improvement: N/A")

        # Reset buttons
        self.btn_add_noise.setEnabled(False)
        self.btn_apply_filter.setEnabled(False)
        self.btn_play_orig.setEnabled(False)
        self.btn_play_noisy.setEnabled(False)
        self.btn_play_filt.setEnabled(False)
        self.btn_save_wav.setEnabled(False)

        # Clear plots
        self.canvas_overview.clear_canvas()
        self.canvas_original.clear_canvas()
        self.canvas_noisy.clear_canvas()
        self.canvas_filtered.clear_canvas()
        self._render_filter_response_canvas()

        self.show_status("Reset complete. Workspace cleared.", "info")

    # ---------------- Plot Rendering ----------------

    def _render_all_plots(self):
        """Render plots across both the overview comparison canvas and individual tab canvases."""
        if self.clean_signal is None:
            return

        # 1. Render Tab 1: Comprehensive 3-Stage Pipeline (Overview)
        self._render_overview_canvas()

        # 2. Render Individual Tabs
        self._render_single_stage(
            self.canvas_original, self.clean_signal, "Original Clean Signal", "#2563EB"
        )
        self._render_filter_response_canvas()
        if self.noisy_signal is not None:
            self._render_single_stage(
                self.canvas_noisy, self.noisy_signal, "Noisy Signal (+AWGN)", "#DC2626"
            )
        else:
            self.canvas_noisy.clear_canvas()

        if self.filtered_signal is not None:
            self._render_single_stage(
                self.canvas_filtered, self.filtered_signal, "Filtered Reconstructed Signal", "#16A34A"
            )
        else:
            self.canvas_filtered.clear_canvas()

    def _render_overview_canvas(self):
        """Render 3 rows x 2 columns on the main comparison canvas."""
        canvas = self.canvas_overview
        fig = canvas.fig
        fig.clear()
        axes = fig.subplots(3, 2)

        stages = [
            ("Original (Clean)", self.clean_signal, "#2563EB"),
            ("Noisy (+AWGN)", self.noisy_signal, "#DC2626"),
            ("Filtered (Butterworth)", self.filtered_signal, "#16A34A"),
        ]

        for row_idx, (title, sig, color) in enumerate(stages):
            ax_time = axes[row_idx, 0]
            ax_freq = axes[row_idx, 1]

            if sig is not None:
                # Downsample for fast, responsive rendering if signal is large
                step = max(1, len(sig) // 20000)
                t = compute_time_axis(len(sig), self.fs)[::step]
                sig_plot = sig[::step]

                # Time domain waveform
                ax_time.plot(t, sig_plot, color=color, linewidth=0.8)
                ax_time.set_title(f"{title} — Time Domain", fontsize=9, fontweight="bold", pad=4)
                ax_time.set_xlabel("Time (seconds)", fontsize=8)
                ax_time.set_ylabel("Amplitude", fontsize=8)
                ax_time.grid(True, linestyle="--", alpha=0.5)
                ax_time.tick_params(labelsize=7)

                # Frequency domain FFT
                freqs, mags = compute_fft(sig, self.fs)
                # Plot frequencies up to 10 kHz or Nyquist
                max_display_f = min(self.fs / 2.0, 10000.0)
                mask = freqs <= max_display_f
                f_plot = freqs[mask]
                m_plot = mags[mask]

                ax_freq.plot(f_plot, m_plot, color=color, linewidth=0.8)
                ax_freq.set_title(f"{title} — FFT Magnitude", fontsize=9, fontweight="bold", pad=4)
                ax_freq.set_xlabel("Frequency (Hz)", fontsize=8)
                ax_freq.set_ylabel("Magnitude", fontsize=8)
                ax_freq.grid(True, linestyle="--", alpha=0.5)
                ax_freq.tick_params(labelsize=7)

                # Annotate peak frequency on FFT plot
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
                    0.5, 0.5, f"{title}\n[Not Yet Applied]",
                    ha="center", va="center", color="#94A3B8", fontsize=9
                )
                ax_time.set_xticks([])
                ax_time.set_yticks([])

                ax_freq.text(
                    0.5, 0.5, f"{title} FFT\n[Not Yet Applied]",
                    ha="center", va="center", color="#94A3B8", fontsize=9
                )
                ax_freq.set_xticks([])
                ax_freq.set_yticks([])

        fig.tight_layout()
        canvas.draw()

    def _render_single_stage(self, canvas: MplCanvas, sig: np.ndarray, title: str, color: str):
        """Render large, high-detail 2-row (Time + FFT) plot on individual stage tabs."""
        fig = canvas.fig
        fig.clear()
        axes = fig.subplots(2, 1)

        ax_time, ax_freq = axes[0], axes[1]

        # Time domain
        step = max(1, len(sig) // 40000)
        t = compute_time_axis(len(sig), self.fs)[::step]
        ax_time.plot(t, sig[::step], color=color, linewidth=1.0)
        ax_time.set_title(f"{title} — Waveform x[n]", fontsize=11, fontweight="bold")
        ax_time.set_xlabel("Time (s)", fontsize=9)
        ax_time.set_ylabel("Amplitude", fontsize=9)
        ax_time.grid(True, linestyle="--", alpha=0.6)

        # Frequency domain
        freqs, mags = compute_fft(sig, self.fs)
        max_display_f = min(self.fs / 2.0, 10000.0)
        mask = freqs <= max_display_f

        ax_freq.plot(freqs[mask], mags[mask], color=color, linewidth=1.0)
        ax_freq.set_title(f"{title} — One-Sided FFT Spectrum |X(f)|", fontsize=11, fontweight="bold")
        ax_freq.set_xlabel("Frequency (Hz)", fontsize=9)
        ax_freq.set_ylabel("Magnitude", fontsize=9)
        ax_freq.grid(True, linestyle="--", alpha=0.6)

        # Annotate peak frequency on single-stage FFT plot
        peak_f, peak_m = find_peak_frequency(freqs, mags)
        if peak_f > 0 and peak_f <= max_display_f:
            ax_freq.axvline(peak_f, color="#E11D48", linestyle=":", linewidth=1.2, alpha=0.8)
            ax_freq.text(
                0.97, 0.92, f"Dominant Peak: {peak_f:.1f} Hz (Mag: {peak_m:.3f})",
                transform=ax_freq.transAxes,
                ha="right", va="top",
                fontsize=9,
                fontweight="bold",
                color="#BE123C",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF1F2", edgecolor="#FECDD3", alpha=0.9),
            )

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

            # Subplot 1: Single-pass Magnitude Response
            ax_mag.plot(freqs, mag_single_db, color="#2563EB", linewidth=1.5, label="Single-Pass |H(f)|")
            ax_mag.axhline(-3.0, color="#E11D48", linestyle="--", linewidth=1.0, label="-3 dB Cutoff")
            ax_mag.set_title(f"Butterworth {f_type.capitalize()} Filter — Single-Pass Magnitude Response", fontsize=10, fontweight="bold")
            ax_mag.set_ylabel("Magnitude (dB)", fontsize=9)
            ax_mag.set_ylim(-80, 5)
            ax_mag.grid(True, linestyle="--", alpha=0.6)
            ax_mag.legend(loc="lower right", fontsize=8)

            # Subplot 2: Effective Zero-Phase Filter Response (sosfiltfilt)
            ax_eff.plot(freqs, mag_eff_db, color="#16A34A", linewidth=1.5, label="Effective Zero-Phase |H_eff(f)| = |H(f)|^2")
            ax_eff.axhline(-6.0, color="#E11D48", linestyle="--", linewidth=1.0, label="-6 dB Effective Point")
            ax_eff.set_title("Effective Zero-Phase Response (sosfiltfilt — Zero Phase Delay)", fontsize=10, fontweight="bold")
            ax_eff.set_xlabel("Frequency (Hz)", fontsize=9)
            ax_eff.set_ylabel("Magnitude (dB)", fontsize=9)
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
        self.status_label.setStyleSheet(f"color: {color}; font-weight: 500; font-size: 9pt;")
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
