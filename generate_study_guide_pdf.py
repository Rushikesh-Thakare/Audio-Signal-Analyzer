"""
generate_study_guide_pdf.py — Generate Academic Study Guide PDF for SNS Project.
Covers all 15 Signals & Systems concepts in beginner-friendly language with
exact codebase mappings and viva answers.
"""

import os
import sys
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
    HRFlowable,
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Canvas for adding page numbers and running header/footer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))

        # Header (Pages 2+)
        if self._pageNumber > 1:
            self.drawString(
                54,
                11 * 72 - 36,
                "Signals & Systems (SNS) Mini Project Study Guide — Audio Signal Analyzer Using FFT",
            )
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, 11 * 72 - 42, 8.5 * 72 - 54, 11 * 72 - 42)

        # Footer (All pages)
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * 72 - 54, 36, page_str)
        self.drawString(
            54,
            36,
            "Confidential & Academic Use Only | Audio Signal Analyzer & Noise Reduction System",
        )
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 46, 8.5 * 72 - 54, 46)
        self.restoreState()


def build_pdf(filename="SNS_Project_Study_Guide.pdf"):
    pdf_path = os.path.abspath(filename)
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    title_style = ParagraphStyle(
        "DocTitle",
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#1E3A8A"),
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#475569"),
        spaceAfter=14,
    )
    h1_style = ParagraphStyle(
        "SectionHeading",
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyDark",
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=5,
    )
    label_style = ParagraphStyle(
        "LabelBold",
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#1E293B"),
    )
    code_style = ParagraphStyle(
        "CodeText",
        fontName="Courier",
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#094191"),
    )
    viva_q_style = ParagraphStyle(
        "VivaQ",
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#92400E"),
    )
    viva_a_style = ParagraphStyle(
        "VivaA",
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1E293B"),
    )

    story = []

    # Title Block
    story.append(Paragraph("Signals & Systems (SNS) Mini Project", subtitle_style))
    story.append(Paragraph("Audio Signal Analyzer & Noise Reduction System", title_style))
    story.append(
        Paragraph(
            "<b>Complete Beginner's Study Guide:</b> 15 Core Concepts Taught From Zero with Code Mappings & Viva Answers",
            subtitle_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563EB"), spaceAfter=14))

    # Content data for all 15 concepts
    concepts = [
        {
            "num": "1",
            "title": "Audio Signal",
            "simple": "Sound is vibration traveling through the air as continuous pressure waves. In computers, an audio signal is represented as numbers indicating how strongly air is being compressed or rarefied at each instant. A pure tone is a smooth repeating wave (sine wave), while real speech or instruments are mixtures of many waves added together.",
            "file": "audio_io.py -> load_audio(filepath) & to_mono(data)",
            "why": "We must import sound files (WAV) into Python and convert them into 1D numerical arrays of floating-point numbers between [-1.0, 1.0] so our math functions can process them.",
            "viva": "What is an audio signal physically and digitally? Physically, an audio signal is a continuous acoustic pressure wave. Digitally, it is a sequence of discrete numerical amplitude values representing that pressure over time.",
        },
        {
            "num": "2",
            "title": "Sampling",
            "simple": "A computer cannot record every infinite millisecond of continuous time. Instead, it takes 'snapshots' (measurements) of sound thousands of times per second—exactly like taking 60 video frames per second to create motion picture. The number of snapshots per second is the Sampling Frequency (fs). Standard CD-quality audio uses fs = 44,100 Hz.",
            "file": "audio_io.py -> load_audio (reads fs) & sample_audio/generate_samples.py",
            "why": "Sampling converts continuous real-world acoustic waves into discrete numbers that a digital computer can store, analyze, and manipulate.",
            "viva": "What is the Nyquist-Shannon Sampling Theorem? It states that the sampling rate fs must be strictly greater than twice the highest frequency present in the signal (fs > 2 * fmax) to prevent distortion known as aliasing.",
        },
        {
            "num": "3",
            "title": "Discrete-Time Signal",
            "simple": "A discrete-time signal is written mathematically as x[n], where n is an integer index (n = 0, 1, 2, ... N-1). Unlike continuous time t which has infinite decimals, n just counts the samples: sample 0, sample 1, sample 2, etc. The actual physical time of sample n is simply t = n / fs.",
            "file": "signal_processing.py -> compute_time_axis(num_samples, fs)",
            "why": "Computers can only perform arithmetic on discrete memory arrays. Our time vector t[n] = n / fs allows us to plot real seconds on graph axes.",
            "viva": "How is continuous time related to sample index n? Continuous time is related by t = n * Ts = n / fs, where Ts is the sampling period and fs is the sampling frequency.",
        },
        {
            "num": "4",
            "title": "Time Domain",
            "simple": "The time domain is the view of a signal where the horizontal axis is Time (seconds) and the vertical axis is Amplitude (volume/voltage). Looking at the time domain shows you WHEN things happen (a drum hit, silence, or sudden loud noise), but it cannot tell you which musical notes or individual pitch frequencies are playing.",
            "file": "gui.py -> _render_waveform_axis(...) & signal_processing.py -> compute_signal_stats()",
            "why": "Visualizing the time-domain waveform lets us verify signal amplitude, spot clipping (> 1.0), inspect envelope shapes, and see random noise fluctuations.",
            "viva": "What are the limitations of analyzing an audio signal purely in the time domain? The time domain only shows amplitude variations over time; it cannot reveal the individual frequency components or musical pitches that compose the signal.",
        },
        {
            "num": "5",
            "title": "Frequency Domain",
            "simple": "The frequency domain decomposes a complex wave into its recipe of pure ingredients. The horizontal axis is Frequency (Hz) and the vertical axis is Magnitude (strength). For example, if you play an A-major chord on a piano, the time domain looks like messy vibrations, but the frequency domain shows three clear vertical spikes corresponding to the notes A, C#, and E.",
            "file": "gui.py -> _render_fft_axis(...) & signal_processing.py -> compute_fft()",
            "why": "Noise and unwanted sounds usually occupy specific frequency ranges. We must look at the frequency domain to pinpoint where the noise lives so we know where to cut it.",
            "viva": "Why do we transform signals into the frequency domain? Because many signal operations—such as identifying pitch, detecting interference tones, and designing frequency-selective filters—are simple and intuitive in the frequency domain but impossible in the time domain.",
        },
        {
            "num": "6",
            "title": "Discrete Fourier Transform (DFT)",
            "simple": "The Discrete Fourier Transform is the exact mathematical equation that converts N samples in the time domain x[n] into N frequency points X[k]. It works by multiplying the signal with complex sine and cosine waves of every possible frequency and adding up the results. If a frequency exists in the audio, the sum is large; if not, it cancels to zero.",
            "file": "signal_processing.py -> compute_fft (mathematical foundation: X[k] = sum(x[n] * e^(-j*2*pi*k*n/N)))",
            "why": "DFT is the formal bridge between discrete time and discrete frequency in computer digital signal processing.",
            "viva": "What is the computational complexity of the direct DFT? Calculating direct DFT requires O(N^2) complex multiplication and addition operations, which is too slow for real-time processing of long audio files.",
        },
        {
            "num": "7",
            "title": "Fast Fourier Transform (FFT)",
            "simple": "The FFT is NOT a different transform; it is an ingenious fast algorithm that calculates the EXACT same DFT in a fraction of the time (using divide-and-conquer). For 100,000 samples, direct DFT requires 10 billion calculations, while FFT takes only about 1.6 million calculations—thousands of times faster!",
            "file": "signal_processing.py -> compute_fft(signal, fs) using np.fft.rfft",
            "why": "We use np.fft.rfft because audio is real-valued. It computes only the positive frequencies up to Nyquist (fs / 2) with normalized physical magnitude.",
            "viva": "Why do we use rfft instead of standard fft? Because real-valued audio signals have symmetric Fourier transforms (Hermitian symmetry). rfft computes only the unique positive half of the spectrum (0 to fs/2), saving 50% CPU time and memory.",
        },
        {
            "num": "8",
            "title": "Noise (AWGN)",
            "simple": "Noise is any unwanted electrical, acoustic, or digital disturbance that corrupts the signal. In this project we use Additive White Gaussian Noise (AWGN): 'Additive' means it simply adds on top of our sound; 'White' means it spreads uniformly across all frequencies like a flat hissing floor; 'Gaussian' means random noise amplitudes follow a standard bell curve.",
            "file": "signal_processing.py -> add_white_gaussian_noise(signal, target_snr_db, seed)",
            "why": "We simulate realistic degradation in a controlled, reproducible manner so we can objectively test how effectively our digital filters remove noise.",
            "viva": "Why is it called 'White' noise? Just as white light contains equal intensities of all visible colors/wavelengths, white noise contains equal power across all frequency bands in the spectrum.",
        },
        {
            "num": "9",
            "title": "Low-Pass Filter (LPF)",
            "simple": "A low-pass filter acts like an acoustic sieve: it lets low frequencies pass through unaffected (bass, human fundamental voice) while blocking and attenuating all high frequencies above a chosen cutoff (high-pitched hiss, whistles, screeching noise).",
            "file": "signal_processing.py -> apply_lowpass_filter(signal, cutoff, fs, order=4)",
            "why": "When high-frequency noise or unwanted high tones (like our 3500 Hz tone) contaminate a low-frequency sound (like our 300 Hz tone), a low-pass filter eliminates the high noise.",
            "viva": "What is the ideal passband and stopband behavior of a low-pass filter? The passband passes all frequencies from 0 to fc with a gain of 1 (0 dB), while the stopband completely eliminates all frequencies above fc with infinite attenuation.",
        },
        {
            "num": "10",
            "title": "High-Pass Filter (HPF)",
            "simple": "The opposite of low-pass: a high-pass filter blocks all low frequencies below the cutoff (such as low AC electrical hum at 50/60 Hz, microphone thumps, wind rumble) while allowing high frequencies to pass through cleanly.",
            "file": "signal_processing.py -> apply_highpass_filter(signal, cutoff, fs, order=4)",
            "why": "Used to remove low-frequency DC offset and low-frequency rumble without altering the treble or high harmonics of speech.",
            "viva": "Give an everyday application of a high-pass filter. Removing low-frequency wind rumble, breathing pops, or 50/60 Hz electrical power-line hum from microphone vocal recordings.",
        },
        {
            "num": "11",
            "title": "Band-Pass Filter (BPF)",
            "simple": "A band-pass filter allows only a specific window of frequencies between a lower cutoff (f_low) and an upper cutoff (f_high) to pass through. It blocks both very low rumble below f_low AND very high hiss above f_high. A classic example is a telephone channel, which only passes speech between 300 Hz and 3400 Hz.",
            "file": "signal_processing.py -> apply_bandpass_filter(signal, low_cutoff, high_cutoff, fs, order=4)",
            "why": "Used when the desired audio signal occupies an intermediate frequency band and is surrounded by both low-end and high-end noise.",
            "viva": "How are the cutoff frequencies of a band-pass filter defined? A band-pass filter has two cutoff frequencies: f_low (lower cutoff) and f_high (upper cutoff), which define the passband bandwidth B = f_high - f_low.",
        },
        {
            "num": "12",
            "title": "Cutoff Frequency (fc)",
            "simple": "The cutoff frequency is the boundary line between the frequencies we keep and the frequencies we discard. On a filter's frequency graph, the cutoff frequency is defined as the point where signal power drops to 50% (which equals an amplitude drop to 70.7%, or -3 dB).",
            "file": "signal_processing.py -> validate_cutoff_frequency(cutoff, fs, filter_type)",
            "why": "Our software strictly checks that 0 < fc < fs / 2 (Nyquist limit). A digital filter cannot have a cutoff frequency above half the sampling rate.",
            "viva": "Why is the cutoff frequency called the -3 dB point? Because -3 dB corresponds to 20 * log10(1 / sqrt(2)) ≈ -3.01 dB, which is the exact frequency where output signal power drops to half of its input power.",
        },
        {
            "num": "13",
            "title": "Butterworth Filter",
            "simple": "A Butterworth filter is known as the 'maximally flat' filter. Other filters (like Chebyshev) have ripples (bumpy waves) in their frequency passband that distort musical harmonics. The Butterworth filter has a completely smooth, flat passband with zero ripple, making it ideal for audio.",
            "file": "signal_processing.py -> design_butterworth_filter(..., output='sos') & sosfiltfilt",
            "why": "We use Second-Order Sections (SOS) to avoid numeric overflow, and zero-phase forward-backward filtering (sosfiltfilt) so audio waveforms experience zero phase delay.",
            "viva": "What is the key advantage of a Butterworth filter, and why do we use zero-phase filtering? Its key advantage is a maximally flat passband with no amplitude ripples. Zero-phase filtering (sosfiltfilt) processes forward and backward to cancel out all phase shifts.",
        },
        {
            "num": "14",
            "title": "Signal-to-Noise Ratio (SNR)",
            "simple": "SNR measures how clean your sound is by comparing desired signal power against unwanted noise power. It is measured in decibels (dB). A high SNR (e.g., 25 dB) means loud clear sound with faint background noise. A low SNR (e.g., 0 dB) means the noise is just as loud as the music. If SNR is negative, the sound is drowned in noise.",
            "file": "signal_processing.py -> calculate_snr(clean_signal, noisy_or_processed_signal)",
            "why": "We calculate SNR Before filtering, SNR After filtering, and Delta SNR (Improvement). This proves mathematically whether our filter actually improved audio quality.",
            "viva": "How is SNR mathematically defined, and what does a positive SNR improvement indicate? SNR(dB) = 10 * log10(P_signal / P_noise). A positive improvement (Delta SNR > 0) proves that our filter successfully reduced noise power relative to the clean signal.",
        },
        {
            "num": "15",
            "title": "Python Architecture & Code Mapping",
            "simple": "The project is cleanly divided into 4 modular layers: 1) audio_io.py handles file reading/saving, stereo-to-mono downmixing, and speaker playback; 2) signal_processing.py performs all the DSP math (FFT, AWGN, Butterworth SOS, SNR); 3) gui.py builds the PySide6 user interface with live Matplotlib graphs; 4) app.py is the launcher.",
            "file": "app.py -> gui.py -> signal_processing.py -> audio_io.py",
            "why": "Separating mathematical logic from the user interface makes the software stable, easy to read for academic evaluation, and 100% unit-testable.",
            "viva": "Walk me through how data flows when the user clicks 'Apply Filter'. The GUI takes noisy_signal, calls signal_processing.apply_filter with the selected filter type and cutoff, runs zero-phase Butterworth filtering, calls calculate_snr against clean_signal, and re-renders the filtered waveform and FFT plots.",
        },
    ]

    for item in concepts:
        card_data = [
            [
                Paragraph(
                    f"<b>Concept {item['num']}: {item['title']}</b>",
                    h1_style,
                )
            ],
            [
                Table(
                    [
                        [
                            Paragraph("<b>Simple Explanation:</b>", label_style),
                            Paragraph(item["simple"], body_style),
                        ],
                        [
                            Paragraph("<b>Project Code Mapping:</b>", label_style),
                            Paragraph(f"<code>{item['file']}</code>", code_style),
                        ],
                        [
                            Paragraph("<b>Why We Use It:</b>", label_style),
                            Paragraph(item["why"], body_style),
                        ],
                        [
                            Paragraph("<b>Viva Exam Q&A:</b>", viva_q_style),
                            Paragraph(f"<b>Q:</b> {item['viva']}", viva_a_style),
                        ],
                    ],
                    colWidths=[120, 384],
                    style=TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("TOPPADDING", (0, 0), (-1, -1), 4),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                            ("LEFTPADDING", (0, 0), (-1, -1), 6),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                        ]
                    ),
                )
            ],
        ]

        concept_table = Table(
            card_data,
            colWidths=[504],
            style=TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                    ("TOPPADDING", (0, 0), (-1, -1), 2),
                ]
            ),
        )
        story.append(KeepTogether([concept_table, Spacer(1, 8)]))

    # Build document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[SUCCESS] Study Guide PDF successfully created: {pdf_path}")
    return pdf_path


if __name__ == "__main__":
    out_file = sys.argv[1] if len(sys.argv) > 1 else "SNS_Project_Study_Guide.pdf"
    build_pdf(out_file)
