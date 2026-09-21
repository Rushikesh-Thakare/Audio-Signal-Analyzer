# Signals and Systems (SNS) Viva Examination Guide

Comprehensive viva preparation guide containing **20 core questions and student-friendly answers** based directly on the actual codebase implementation of the **Audio Signal Analyzer and Noise Reduction System Using FFT**.

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
  For an audio file sampled at $f_s = 44100\text{ Hz}$, the maximum frequency that can be analyzed or filtered is $22050\text{ Hz}$. In `signal_processing.py` (`validate_cutoff_frequency`), our code strictly validates that any filter cutoff frequency $f_c$ satisfies $0 < f_c < f_s / 2$.

---

### Q3: What is aliasing, and how is it prevented in digital audio?
**Answer:**
- **Aliasing:** If a signal contains frequencies higher than $f_s / 2$, those high frequencies fold back (reflect) across the Nyquist boundary and appear as false, distorted low frequencies in the digitized signal.
- **Prevention:** An analog low-pass filter (called an **anti-aliasing filter**) is placed before the Analog-to-Digital Converter (ADC) to remove all frequencies above $f_s / 2$ prior to sampling.

---

### Q4: How do we downmix stereo audio to mono, and why is it necessary for this project?
**Answer:**
- A stereo recording has two separate channels (Left and Right, $x_L[n]$ and $x_R[n]$).
- For classical 1D linear signal processing and single-channel FFT analysis, we convert stereo to mono by arithmetic averaging across channels:
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
  1. We ignore the $0\text{ Hz}$ DC component by searching frequencies above $f > 0$.
  2. We find the index of the maximum magnitude: $k_{\text{peak}} = \arg\max(\text{Magnitude})$.
  3. The peak frequency is $f_{\text{peak}} = \text{freqs}[k_{\text{peak}}]$.
- In the GUI, this peak frequency is dynamically annotated and marked with a vertical red dashed line.

---

## Section 3: Additive White Gaussian Noise (AWGN)

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
- **In `signal_processing.py` (`add_white_gaussian_noise`):**
  We generate standard normal random variables using `np.random.normal(0.0, 1.0, N)` and scale them by $\sigma = \sqrt{\sigma^2}$.

---

### Q12: Why is the clean signal kept immutable when noise is added?
**Answer:**
- In academic DSP and software design, original data must never be overwritten destructively.
- Maintaining separate arrays (`clean_signal`, `noisy_signal`, `noise_component`) allows:
  1. Objective ground-truth mathematical error calculation: $e[n] = y[n] - x[n]$.
  2. Side-by-side comparative visualization in the GUI.
  3. Independent audio playback of both clean and noisy versions.

---

## Section 4: Digital Filters & Frequency Response

### Q13: What is a Linear Time-Invariant (LTI) system, and why are filters LTI systems?
**Answer:**
- **Linearity:** The system satisfies superposition. If input $x_1[n] \to y_1[n]$ and $x_2[n] \to y_2[n]$, then:
  $$a x_1[n] + b x_2[n] \to a y_1[n] + b y_2[n]$$
- **Time-Invariance:** Shifting the input in time simply shifts the output by the same amount:
  $$x[n - n_0] \to y[n - n_0]$$
- Digital Butterworth filters are LTI systems governed by constant-coefficient linear difference equations.

---

### Q14: What is the difference between an FIR and an IIR filter?
**Answer:**
- **FIR (Finite Impulse Response):** The impulse response is finite in duration. FIR filters use only present and past input samples (no feedback loop). They are always stable and can easily have linear phase, but require high filter orders (many coefficients) to achieve sharp roll-offs.
- **IIR (Infinite Impulse Response):** The impulse response theoretically continues indefinitely because the filter uses feedback (past output samples as well as inputs):
  $$y[n] = \sum_{k=0}^{M} b_k x[n-k] - \sum_{k=1}^{P} a_k y[n-k]$$
  IIR filters achieve very steep frequency roll-offs with much lower order (e.g. Order 4), requiring significantly less memory and computation.

---

### Q15: Why did we choose a Butterworth filter over Chebyshev or Elliptic filters?
**Answer:**
- **Butterworth Filter:** Known as the **maximally flat** magnitude filter.
  - In the passband: The frequency response is completely flat with zero passband ripple.
  - In the stopband: It rolls off monotonically toward zero.
- **Chebyshev / Elliptic filters** have steeper transition bands, but they introduce ripples (oscillations in magnitude) in the passband or stopband, which distorts audio harmonics. For high-fidelity audio, a smooth Butterworth response is preferred.

---

### Q16: What are Second-Order Sections (SOS), and why does our code use them instead of `(b, a)`?
**Answer:**
- A 4th-order IIR filter can be expressed as a single high-degree rational transfer function $H(z) = B(z) / A(z)$.
- However, with higher orders, polynomial roots (poles and zeros) become extremely sensitive to finite 64-bit floating-point rounding errors. High polynomial orders can cause poles to drift outside the unit circle, resulting in catastrophic numerical overflow or oscillation.
- **Second-Order Sections (SOS):** The 4th-order filter is factored into a cascade of two stable 2nd-order biquad filter sections:
  $$H(z) = H_1(z) \cdot H_2(z)$$
- In `signal_processing.py`, we use `scipy.signal.butter(..., output='sos')`. This guarantees numerical stability across all sample rates and cutoff frequencies.

---

### Q17: What is the $-3\text{ dB}$ cutoff point on a filter frequency response?
**Answer:**
- The **cutoff frequency ($f_c$)** is the half-power frequency.
- In decibels:
  $$20 \log_{10}\left(\frac{1}{\sqrt{2}}\right) \approx -3.0103\text{ dB}$$
- At frequency $f_c$, the output signal voltage/amplitude drops to $70.7\%$ ($1/\sqrt{2}$) of the passband input, which corresponds to exactly $50\%$ ($1/2$) signal power.

---

### Q18: What is phase distortion, and how does zero-phase filtering (`sosfiltfilt`) eliminate it?
**Answer:**
- **Phase Distortion / Group Delay:** Standard causal digital filtering delays different frequency components by different amounts of time, causing audio waveforms to smear or disperse.
- **Zero-Phase Forward-Backward Filtering (`scipy.signal.sosfiltfilt`):**
  1. The audio is filtered forward: $x[n] \xrightarrow{H(z)} y_{\text{fwd}}[n]$. The output picks up a phase shift $\theta(\omega)$.
  2. The output is reversed in time and filtered through the exact same filter backward: $y_{\text{fwd}}[-n] \xrightarrow{H(z)} y_{\text{rev}}[n]$. This introduces an opposite phase shift $-\theta(\omega)$.
  3. The result is reversed again.
- **Mathematical cancellation:**
  $$\theta(\omega) + (-\theta(\omega)) = 0$$
  The net phase response is identically zero radians for all frequencies, preserving the sharp transient timing of the original waveform! The effective filter order is doubled ($4 \times 2 = 8$), resulting in a steep $-48\text{ dB/octave}$ roll-off.

---

## Section 5: Signal-to-Noise Ratio (SNR)

### Q19: How is SNR mathematically calculated when a ground-truth reference exists?
**Answer:**
- Given clean ground-truth signal $x[n]$ and processed/noisy signal $y[n]$, the error sequence is defined as:
  $$e[n] = y[n] - x[n]$$
- Signal power: $P_{\text{clean}} = \frac{1}{N} \sum_{n=0}^{N-1} x[n]^2$
- Noise/Error power: $P_{\text{error}} = \frac{1}{N} \sum_{n=0}^{N-1} (y[n] - x[n])^2$
- Signal-to-Noise Ratio in decibels:
  $$\text{SNR}_{\text{dB}} = 10 \log_{10}\left( \frac{P_{\text{clean}}}{P_{\text{error}}} \right) = 10 \log_{10}\left( \frac{\sum x[n]^2}{\sum (y[n] - x[n])^2} \right)$$
- **In our project:**
  - $\text{SNR}_{\text{before}}$ evaluates $y = x + \text{noise}$.
  - $\text{SNR}_{\text{after}}$ evaluates $y = \text{filtered\_signal}$.
  - **SNR Gain:** $\Delta\text{SNR} = \text{SNR}_{\text{after}} - \text{SNR}_{\text{before}}$. A positive value confirms objective noise attenuation.

---

### Q20: Why can't we calculate ground-truth SNR on arbitrary real-world recordings?
**Answer:**
- Mathematical SNR requires access to the pure clean signal $x[n]$ to compute the denominator error power $\sum (y[n] - x[n])^2$.
- When a user records audio in a noisy room or loads an unknown audio track from the internet, the recording already contains unseparated noise. Without the original clean audio recorded simultaneously in an anechoic chamber, the true error cannot be known with certainty.
- Fabricating an SNR without a ground-truth reference is scientifically dishonest. In our system, whenever an arbitrary external WAV is loaded without adding synthetic noise, the SNR display honestly shows **"N/A"** until a known controlled noise reference is injected.
