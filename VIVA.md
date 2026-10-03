# Signals and Systems (SNS) Viva Examination Guide

Comprehensive viva preparation guide containing **25 core questions and student-friendly answers** based directly on the actual codebase implementation of the **Audio Signal Analyzer and Noise Reduction System Using FFT**.

---

## Section 1: Sampling & Discrete-Time Representation

### Q1: What is continuous-time audio vs. discrete-time audio, and how is sound digitized?
**Answer:**
- **Continuous-time audio ($x(t)$):** Sound in the physical world is an analog, continuous pressure wave that exists at every infinite instant of time $t \in \mathbb{R}$.
- **Discrete-time audio ($x[n]$):** Computers cannot store infinite points, so we take samples at uniform discrete intervals $T_s = 1/f_s$. The sequence is indexed by an integer $n \in \{0, 1, 2, \dots, N-1\}$.
- **In our code (`audio_io.py` & `signal_processing.py`):**
  When a WAV file is loaded via `soundfile.read()`, continuous sound is represented as a 1D NumPy array `x` of sample values floating between $[-1.0, 1.0]$. The time vector is computed as:
  $$t[n] = \frac{n}{f_s}$$
  where $f_s$ is the sampling frequency in Hertz (e.g., $44100\text{ Hz}$ or $16000\text{ Hz}$).

---

### Q2: What is the Nyquist-Shannon Sampling Theorem, and what is the Nyquist limit?
**Answer:**
- The **Nyquist-Shannon Sampling Theorem** states that to perfectly reconstruct a continuous band-limited signal without distortion, the sampling frequency $f_s$ must be strictly greater than twice the highest frequency component present in the signal ($f_{\text{max}}$):
  $$f_s > 2 \cdot f_{\text{max}}$$
- **Nyquist Frequency ($f_{\text{Nyquist}}$):** Half the sampling rate:
  $$f_{\text{Nyquist}} = \frac{f_s}{2}$$
- **In our project:**
  For an audio file sampled at $f_s = 44100\text{ Hz}$, the maximum frequency that can be analyzed or filtered is $22050\text{ Hz}$. In `signal_processing.py` (`validate_filter_cutoffs`), our code strictly validates that any filter cutoff frequency $f_c$ satisfies $0 < f_c < f_s / 2$.

---

### Q3: What is aliasing, and how is it prevented in digital audio?
**Answer:**
- **Aliasing:** If a signal contains frequencies higher than $f_s / 2$, those high frequencies fold back (reflect) across the Nyquist boundary and appear as false, distorted low frequencies in the digitized signal.
- **Prevention:** An analog low-pass filter (called an **anti-aliasing filter**) is placed before the Analog-to-Digital Converter (ADC) to remove all frequencies above $f_s / 2$ prior to sampling.

---

### Q4: How do we downmix stereo audio to mono, and why is it necessary for this project?
**Answer:**
- A stereo recording has two separate channels (Left and Right, $x_L[n]$ and $x_R[n]$).
- For classical 1D linear signal processing, speech analysis, and single-channel FFT computation, we downmix stereo to mono by arithmetic averaging across channels:
  $$x_{\text{mono}}[n] = \frac{x_L[n] + x_R[n]}{2}$$
- **In our code (`audio_io.py` -> `to_mono`):**
  We check `audio_data.ndim == 2` and compute `np.mean(audio_data, axis=1)`. This prevents channel phase cancellation artifacts while ensuring energy is preserved symmetrically.

---

## Section 2: Fast Fourier Transform (FFT) & Frequency Spectrum

### Q5: What is the difference between Fourier Transform (FT), DFT, and FFT?
**Answer:**
1. **Fourier Transform (FT):** Continuous-time, continuous-frequency integral:
   $$X(\Omega) = \int_{-\infty}^{\infty} x(t) e^{-j\Omega t} dt$$
2. **Discrete Fourier Transform (DFT):** Discrete-time, discrete-frequency mathematical summation for finite sequences of length $N$:
   $$X[k] = \sum_{n=0}^{N-1} x[n] e^{-j \frac{2\pi}{N} k n}, \quad k = 0, 1, \dots, N-1$$
   Computing the DFT directly requires $\mathcal{O}(N^2)$ complex operations.
3. **Fast Fourier Transform (FFT):** An efficient algorithm (such as the Cooley-Tukey divide-and-conquer algorithm) that computes the exact same DFT in only $\mathcal{O}(N \log_2 N)$ operations. For $N = 100,000$, FFT is thousands of times faster than standard DFT.

---

### Q6: Why does our project use `np.fft.rfft` instead of standard `np.fft.fft`?
**Answer:**
- Real-world audio signals are strictly real-valued ($x[n] \in \mathbb{R}$, zero imaginary component).
- The Fourier transform of any real signal exhibits **Hermitian conjugate symmetry**:
  $$X[-k] = X^*[k]$$
  This means the negative frequency spectrum is an exact mirror copy of the positive frequency spectrum.
- Standard `fft` calculates all $N$ complex frequency bins (both positive and negative).
- `np.fft.rfft` computes only the non-redundant one-sided spectrum from $0\text{ Hz}$ (DC) up to the Nyquist frequency $f_s/2$ (a total of $N/2 + 1$ bins). This saves 50% computation and memory while matching physical reality.

---

### Q7: How is the magnitude spectrum normalized in our codebase?
**Answer:**
- In `signal_processing.py` (`compute_fft`), the raw complex output of `np.fft.rfft` is normalized so that the physical amplitude matches the time-domain waveform:
  1. The raw complex bins are scaled by the length of the signal: $|X[k]| / N$.
  2. Because negative frequencies are folded into the positive half, all AC bins ($k > 0$) are multiplied by 2:
     $$\text{Magnitude}[k] = \frac{2 \cdot |X[k]|}{N}$$
  3. The DC bin ($k = 0$) and the Nyquist bin ($k = N/2$) are kept at $1/N$.
- **Result:** A sine wave $x[n] = 0.5 \sin(2\pi \cdot 440 \cdot t)$ will produce a distinct spike with a peak height of exactly $0.5$ at $440\text{ Hz}$ on our plot.

---

### Q8: What is frequency resolution ($\Delta f$), and what determines it?
**Answer:**
- **Frequency resolution ($\Delta f$):** The smallest frequency difference between two adjacent frequency bins in the FFT:
  $$\Delta f = \frac{f_s}{N} = \frac{1}{T_{\text{total}}}$$
  where $N$ is the number of samples, and $T_{\text{total}}$ is the total duration of the audio recording in seconds.
- **Key Takeaway:** To distinguish two very close frequencies (e.g. $440\text{ Hz}$ and $442\text{ Hz}$), you must record audio for a longer duration ($T_{\text{total}} \ge 0.5\text{ s}$). Sampling faster ($f_s$) increases the highest observable frequency (Nyquist), but only recording longer ($T$) increases spectral resolution.

---

### Q9: How does our project locate the dominant peak frequency?
**Answer:**
- In `signal_processing.py` (`find_peak_frequency`):
  1. We ignore the sub-audible drift and $0\text{ Hz}$ DC component by searching frequencies above $f \ge 20\text{ Hz}$.
  2. We find the index of the maximum magnitude: $k_{\text{peak}} = \arg\max(\text{Magnitude})$.
  3. The peak frequency is $f_{\text{peak}} = \text{freqs}[k_{\text{peak}}]$.
- In the GUI, this peak frequency is dynamically annotated and marked with a vertical dashed line.

---

## Section 3: Additive White Gaussian Noise (AWGN) & Academic Benchmarking

### Q10: What is Additive White Gaussian Noise (AWGN), and why is it called "White" and "Gaussian"?
**Answer:**
- **Additive:** The noise is added directly to the clean signal:
  $$y[n] = x[n] + w[n]$$
- **White:** Like white light containing all visible colors, white noise contains equal power across all frequency bands (flat Power Spectral Density).
- **Gaussian:** The amplitude of the noise samples at any given instant follows a Gaussian (normal) probability distribution:
  $$w[n] \sim \mathcal{N}(\mu, \sigma^2), \quad \mu = 0$$

---

### Q11: How do we compute the exact noise variance $\sigma^2$ from a target SNR in decibels?
**Answer:**
- Signal power is defined as the mean square of the clean samples:
  $$P_x = \frac{1}{N} \sum_{n=0}^{N-1} x[n]^2$$
- Desired Signal-to-Noise Ratio (linear ratio $R$) is converted from decibels:
  $$\text{SNR}_{\text{dB}} = 10 \log_{10}(R) \implies R = 10^{\frac{\text{SNR}_{\text{dB}}}{10}} = \frac{P_x}{P_{\text{noise}}}$$
- Since the mean of the noise is zero ($\mu = 0$), noise power equals noise variance ($P_{\text{noise}} = \sigma^2$):
  $$\sigma^2 = \frac{P_x}{10^{\frac{\text{target\_SNR}_{\text{dB}}}{10}}}$$
- In `signal_processing.py` (`add_white_gaussian_noise`), we generate normal random variables and scale them by $\sigma = \sqrt{\sigma^2}$.

---

### Q12: Why is artificial noise injection restricted to the Academic Lab Benchmark Tool?
**Answer:**
- The primary real-world purpose of the system (and for the LINA Linux voice assistant) is to take **already-noisy** audio and produce **clean** speech.
- Adding artificial noise to clean audio is an inverted workflow. We retain artificial AWGN injection exclusively as a secondary "Academic Lab Benchmark Tool" so students and professors can verify mathematical $\Delta\text{SNR}$ improvement against an authentic known clean ground truth.

---

## Section 4: Digital Butterworth Filters & SOS

### Q13: What is a Linear Time-Invariant (LTI) system, and why are filters LTI systems?
**Answer:**
- **Linearity:** Superposition holds. If $x_1[n] \to y_1[n]$ and $x_2[n] \to y_2[n]$, then $a x_1[n] + b x_2[n] \to a y_1[n] + b y_2[n]$.
- **Time-Invariance:** Shifting input shifts output identically: $x[n - n_0] \to y[n - n_0]$.
- Digital Butterworth filters are LTI systems described by constant-coefficient linear difference equations.

---

### Q14: What is the difference between an FIR and an IIR filter?
**Answer:**
- **FIR (Finite Impulse Response):** The impulse response has finite duration. FIR filters use only present and past inputs (no feedback loop). They are always stable and can have exact linear phase, but require high orders (e.g. Order 64–256) to achieve sharp cutoff slopes.
- **IIR (Infinite Impulse Response):** Uses feedback (past output samples as well as inputs):
  $$y[n] = \sum_{k=0}^{M} b_k x[n-k] - \sum_{k=1}^{P} a_k y[n-k]$$
  IIR filters achieve steep frequency attenuation slopes with very low orders (e.g. Order 4), requiring substantially less memory and computational overhead.

---

### Q15: Why did we choose a Butterworth filter over Chebyshev or Elliptic filters?
**Answer:**
- **Butterworth Filter:** Known as the **maximally flat** magnitude filter. It has zero passband ripple and rolls off monotonically toward zero in the stopband.
- **Chebyshev / Elliptic filters** have steeper transition bands, but they introduce ripples (oscillations in gain) in the passband or stopband, which distorts musical harmonics and speech vowel formants. For audio fidelity, maximally flat Butterworth is preferred.

---

### Q16: What are Second-Order Sections (SOS), and why does our code use them instead of `(b, a)`?
**Answer:**
- A high-order IIR filter expressed as a single rational transfer function $H(z) = B(z) / A(z)$ suffers from extreme coefficient quantization sensitivity. Small rounding errors in floating-point math can displace poles outside the unit circle, causing instability.
- **Second-Order Sections (SOS):** The 4th-order filter is factored into a cascade of two stable 2nd-order biquad filter sections:
  $$H(z) = H_1(z) \cdot H_2(z)$$
- In `signal_processing.py`, we use `scipy.signal.butter(..., output='sos')`. This guarantees numerical stability across all cutoff frequencies.

---

### Q17: What is phase distortion, and how does zero-phase filtering (`sosfiltfilt`) eliminate it?
**Answer:**
- **Phase Distortion / Group Delay:** Standard causal filtering delays different frequencies by different time amounts, smearing transient attacks.
- **Zero-Phase Forward-Backward Filtering (`sosfiltfilt`):**
  1. Filter audio forward: introduces phase shift $+\theta(\omega)$.
  2. Reverse signal in time and filter backward: introduces opposite phase shift $-\theta(\omega)$.
  3. Reverse the signal back to normal time.
- Net phase delay: $\theta(\omega) - \theta(\omega) \equiv 0$ radians across all frequencies.

---

## Section 5: Speech Spectral Denoising & LINA Voice Assistant

### Q18: Why can't a Butterworth filter remove background room noise from speech?
**Answer:**
- Butterworth filters are linear time-invariant (LTI) frequency-band filters. They can only remove noise in frequency bands *outside* the passband.
- Ambient room noise (computer cooling fans, air conditioning hum, microphone pre-amp thermal hiss) is **broadband**; it occupies the exact same frequencies as speech (300 Hz to 3400 Hz).
- Low-passing speech cuts off high consonants ('s', 't', 'f') making speech unintelligible. Band-passing still leaves in-band noise untouched.
- Therefore, speech denoising requires **time-frequency spectral gating** via the Short-Time Fourier Transform (STFT).

---

### Q19: What is the Short-Time Fourier Transform (STFT), and how does it differ from a standard FFT?
**Answer:**
- Standard FFT assumes the signal is stationary and averages frequency content over the *entire* recording duration, losing all temporal timing information.
- **STFT:** Decomposes a non-stationary signal (like human speech) into small overlapping windowed time slices (e.g. 25–35 ms frames with 75% overlap using a Hann window):
  $$X(f, t) = \sum_{n} x[n] w[n - t] e^{-j 2\pi f n}$$
- This generates a 2D time-frequency matrix (Spectrogram) showing *which* frequencies are active *at which specific millisecond*.

---

### Q20: How does the spectral gating algorithm in our project work?
**Answer:**
1. **STFT Decomposition:** Transform noisy signal into magnitude $|Y(f, t)|$ and phase $\phi(f, t)$.
2. **Noise Profile Estimation:** Calculate average noise magnitude $\mu_{\text{noise}}(f)$ and variation $\sigma_{\text{noise}}(f)$ from a stationary noise interval to set threshold $T(f) = \mu + 1.2\sigma$.
3. **Soft-Knee Wiener Gain Mask:** Rather than hard binary gating (which creates harsh musical noise), compute a smooth Wiener-type gain:
   $$G(f, t) = \frac{1}{1 + \left(\frac{\alpha \cdot T(f)}{|Y(f, t)| + \epsilon}\right)^2}$$
   where $\alpha$ is the user-controlled reduction strength.
4. **Spectral Floor ($\beta$):** Enforce $G(f, t) \ge \beta$ (where $\beta = 0.05 = -26\text{ dB}$) so background pauses do not drop to unnatural digital vacuum silence.
5. **Temporal Inter-Frame Smoothing:** Smooth gain factors across adjacent time frames to suppress isolated single-frame musical noise spikes.
6. **ISTFT Re-synthesis:** Multiply magnitude by gain mask and reconstruct time-domain audio using the original phase: $\hat{x}[n] = \text{ISTFT}(G(f, t) \cdot |Y(f, t)| e^{j\phi(f, t)})$.

---

### Q21: How is the noise profile sample selected, and why don't we assume the first 0.3s is noise?
**Answer:**
- **Why not blind assumption:** In voice assistants (like LINA), users often begin speaking the millisecond they trigger recording. Blindly sampling the first 0.3 seconds would sample vocal formants as "noise", causing the algorithm to aggressively cancel the user's voice!
- **Our Intelligent Method (`detect_noise_segment`):**
  1. Scans the recording using sliding analysis windows to find the global minimum-energy (quietest) window.
  2. Compares the initial 0.3s window to this minimum. If the initial window is within 3 dB of the quietest window, the leading pause is selected.
  3. If the user spoke immediately at $t=0$, the algorithm automatically selects the quietest inter-word pause elsewhere in the clip.
  4. Users can also manually specify exact start and end times in the GUI.

---

### Q22: What is "musical noise", and how does our implementation prevent it?
**Answer:**
- **Musical Noise:** Annoying metallic, chirp-like musical artifacts that occur when spectral subtraction algorithms randomly zero out or keep isolated frequency bins from frame to frame due to statistical noise variance.
- **Our Prevention Safeguards:**
  1. **Soft-Knee Mask:** Uses a smooth Wiener curve instead of harsh binary cutoff.
  2. **Spectral Floor:** Sets a $-26\text{ dB}$ minimum attenuation limit so bins never drop to absolute zero.
  3. **Temporal Recursive Smoothing:** Blends the gain mask with the previous frame:
     $$G_{\text{smooth}}(f, t) = 0.35 \cdot G(f, t) + 0.65 \cdot G_{\text{smooth}}(f, t-1)$$
     This prevents abrupt single-frame energy fluctuations.

---

### Q23: How does the LINA Linux Voice Assistant integrate with this system?
**Answer:**
- In `signal_processing.py`, we provide a standalone, decoupled function:
  ```python
  from signal_processing import denoise_audio
  cleaned_speech = denoise_audio(raw_mic_samples, fs=16000, strength=0.75)
  ```
- This function has zero dependencies on PySide6, Matplotlib, or the desktop GUI.
- LINA invokes this function immediately after microphone capture as an acoustic pre-processor before feeding the audio to speech-to-text models (such as Whisper or Vosk), significantly improving wake-word detection and intent recognition accuracy.

---

## Section 6: Honest Metrics & Scientific Integrity

### Q24: Why is Ground-Truth SNR reported as N/A on real audio recordings?
**Answer:**
- By mathematical definition:
  $$\text{SNR}_{\text{dB}} = 10 \log_{10}\left( \frac{\sum x_{\text{clean}}[n]^2}{\sum (x_{\text{processed}}[n] - x_{\text{clean}}[n])^2} \right)$$
- In a real field recording, no clean ground truth $x_{\text{clean}}[n]$ exists in memory.
- Comparing the cleaned audio to the *noisy input* does not calculate SNR; it only measures how much the signal changed. Claiming that as "SNR" is mathematically false.
- Academic integrity demands displaying **"N/A"** for ground-truth SNR on real recordings.

---

### Q25: What objective metrics are used for real-world unreferenced recordings?
**Answer:**
Our system computes physically verifiable measurements:
1. **Noise Floor Attenuation ($\text{dB}$):**
   $$\Delta \text{Noise} = 20 \log_{10}\left( \frac{\text{RMS}_{\text{noisy, quiet\_zone}}}{\text{RMS}_{\text{cleaned, quiet\_zone}} + \epsilon} \right)$$
   Measures how many decibels the stationary background noise was attenuated.
2. **Speech Energy Retention Ratio ($\%$):**
   $$\text{Retention} = \frac{\text{RMS}_{\text{cleaned, active\_speech}}}{\text{RMS}_{\text{noisy, active\_speech}} + \epsilon} \times 100\%$$
   Verifies that speech energy is preserved (typically $>90\%$), proving that the denoiser did not simply mute the entire recording.
3. **Overall RMS Level Change ($\text{dB}$):**
   Net energy change across the entire recording.
