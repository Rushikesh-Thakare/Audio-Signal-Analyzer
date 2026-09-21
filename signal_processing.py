"""
signal_processing.py — Signals and Systems DSP Core Module

This module implements the fundamental discrete-time signal processing operations:
1. Time-domain representation and time-axis calculation.
2. Signal statistics (peak amplitude, DC offset / mean, RMS energy).
3. Fast Fourier Transform (FFT) analysis for real-valued audio signals.
4. Frequency-axis generation and one-sided magnitude spectrum normalization.
5. Dominant peak frequency detection.

Signals & Systems Principles:
- Sampling: Continuous audio x(t) is sampled at interval Ts = 1/fs to produce discrete sequence x[n].
- Time Vector: t[n] = n * Ts = n / fs for n = 0, 1, ..., N-1.
- DFT/FFT: Transforms time-domain sequence x[n] into frequency components X[k].
- Symmetry: For real signals x[n] in R, X[-k] = X*[k]. The one-sided FFT (rfft) discards redundant
  negative frequencies and scales AC amplitudes by 2 to preserve physical amplitude.
"""

from typing import Tuple, Dict, Any, Optional, Union, List
import numpy as np
import scipy.signal


def compute_time_axis(num_samples: int, fs: int) -> np.ndarray:
    """
    Generate the discrete-time axis array in seconds for a given sample count and sampling rate.

    SNS Formula:
        t[n] = n / fs,  for n in [0, 1, 2, ..., N - 1]

    Args:
        num_samples: Total number of discrete samples N.
        fs: Sampling frequency in Hertz (samples per second).

    Returns:
        np.ndarray: 1D array of time values in seconds from 0 to (N-1)/fs.

    Raises:
        ValueError: If num_samples is negative or fs is <= 0.
    """
    if num_samples < 0:
        raise ValueError(f"Number of samples cannot be negative: {num_samples}")
    if fs <= 0:
        raise ValueError(f"Sampling frequency must be strictly positive: {fs}")

    if num_samples == 0:
        return np.array([], dtype=np.float64)

    return np.arange(num_samples, dtype=np.float64) / float(fs)


def compute_rms(signal: np.ndarray) -> float:
    """
    Calculate the Root Mean Square (RMS) value of a discrete-time signal.

    SNS Formula:
        x_RMS = sqrt((1 / N) * sum_{n=0}^{N-1} x[n]^2)

    RMS represents the effective value or square-root of the average power of the signal.

    Args:
        signal: 1D numpy array of signal values.

    Returns:
        float: RMS value. Returns 0.0 for an empty signal.
    """
    if signal.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(signal, dtype=np.float64))))


def compute_signal_stats(signal: np.ndarray) -> Dict[str, float]:
    """
    Compute key time-domain characteristics and statistics of the signal.

    Args:
        signal: 1D numpy array of signal samples.

    Returns:
        dict containing:
            - 'samples': Total sample count N
            - 'min': Minimum sample value
            - 'max': Maximum sample value
            - 'peak': Peak absolute amplitude max(|x[n]|)
            - 'mean': DC component / average value
            - 'rms': Root Mean Square value
    """
    if signal.size == 0:
        return {
            "samples": 0,
            "min": 0.0,
            "max": 0.0,
            "peak": 0.0,
            "mean": 0.0,
            "rms": 0.0,
        }

    return {
        "samples": int(signal.size),
        "min": float(np.min(signal)),
        "max": float(np.max(signal)),
        "peak": float(np.max(np.abs(signal))),
        "mean": float(np.mean(signal)),
        "rms": compute_rms(signal),
    }


def compute_fft(signal: np.ndarray, fs: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the one-sided Fast Fourier Transform (FFT) magnitude spectrum of a real-valued signal.

    SNS Theory:
    For a real-valued signal x[n] of length N:
    1. The discrete Fourier Transform X[k] possesses Hermitian symmetry: X[N-k] = X*[k].
    2. Therefore, all spectral information is contained in frequencies [0, fs/2] (up to Nyquist).
    3. We use `np.fft.rfft`, which computes only the non-negative frequency bins (N // 2 + 1 points).
    4. Magnitude Normalization:
       - DC bin (k = 0): magnitude = |X[0]| / N
       - AC bins (0 < k < Nyquist): magnitude = 2 * |X[k]| / N
         (Factor of 2 accounts for energy folded from negative frequencies)
       - Nyquist bin (if N is even, k = N/2): magnitude = |X[N/2]| / N

    With this normalization, a pure sine wave A * sin(2 * pi * f0 * t) has a spectral peak
    magnitude exactly equal to A.

    Args:
        signal: 1D numpy array of real-valued signal samples.
        fs: Sampling rate in Hz.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - frequencies: 1D array of frequency bins in Hz from 0 to fs/2.
            - magnitude: 1D array of normalized physical amplitudes.

    Raises:
        ValueError: If signal is empty or fs <= 0.
    """
    if signal.size == 0:
        raise ValueError("Cannot compute FFT of an empty signal.")
    if fs <= 0:
        raise ValueError(f"Sampling frequency must be positive: {fs}")

    n = len(signal)

    # 1. Compute one-sided Fast Fourier Transform for real signal
    fft_complex = np.fft.rfft(signal)

    # 2. Compute the corresponding frequency bins [0, fs/2]
    frequencies = np.fft.rfftfreq(n, d=1.0 / fs)

    # 3. Compute normalized magnitude
    # Base normalization divides by N
    magnitude = np.abs(fft_complex) / float(n)

    # Scale AC components by 2 (all bins except DC at index 0)
    if len(magnitude) > 1:
        magnitude[1:] *= 2.0

        # If N is even, the Nyquist frequency at the final index does not have a negative pair
        if n % 2 == 0:
            magnitude[-1] /= 2.0

    return frequencies, magnitude


def find_peak_frequency(
    frequencies: np.ndarray, magnitude: np.ndarray, min_freq: float = 20.0
) -> Tuple[float, float]:
    """
    Find the dominant peak frequency and its magnitude in a spectrum.

    Args:
        frequencies: 1D array of frequency values in Hz.
        magnitude: 1D array of magnitude values.
        min_freq: Minimum frequency threshold in Hz to exclude DC offset and sub-audible drift.

    Returns:
        Tuple[float, float]: (peak_frequency_hz, peak_magnitude)
    """
    if len(frequencies) == 0 or len(magnitude) == 0:
        return 0.0, 0.0

    # Filter frequencies >= min_freq to ignore DC components
    valid_mask = frequencies >= min_freq
    if not np.any(valid_mask):
        # Fallback to absolute max if all frequencies are below min_freq
        idx = int(np.argmax(magnitude))
        return float(frequencies[idx]), float(magnitude[idx])

    sub_freqs = frequencies[valid_mask]
    sub_mags = magnitude[valid_mask]

    peak_idx = int(np.argmax(sub_mags))
    return float(sub_freqs[peak_idx]), float(sub_mags[peak_idx])


def add_white_gaussian_noise(
    signal: np.ndarray,
    target_snr_db: Optional[float] = None,
    noise_factor: Optional[float] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add zero-mean Additive White Gaussian Noise (AWGN) to a signal.

    SNS Theory:
    - White Gaussian Noise w[n] has a constant power spectral density across all frequencies
      and amplitude samples following a normal distribution w[n] ~ N(0, sigma^2).
    - Signal Power P_signal is given by:
          P_signal = (1 / N) * sum_{n=0}^{N-1} x[n]^2
    - Target Signal-to-Noise Ratio (SNR) in decibels:
          SNR_dB = 10 * log10(P_signal / P_noise)
      Rearranging to solve for the required noise variance sigma^2:
          P_noise = sigma^2 = P_signal / (10^(target_snr_db / 10))
          sigma = sqrt(P_noise)
    - If noise_factor is provided instead (e.g. 0.05):
          sigma = noise_factor * peak_amplitude

    Guarantees:
    - The input clean array is preserved completely unchanged.
    - Returns a new noisy signal array and the isolated noise component.
    - Random generator uses an independent Generator with optional seed for strict reproducibility.

    Args:
        signal: 1D numpy array of clean audio samples x[n].
        target_snr_db: Desired SNR in decibels (e.g. 20.0 for light noise, 0.0 for heavy noise).
        noise_factor: Direct amplitude scale factor if target_snr_db is None.
        seed: Optional integer seed for repeatable noise sequences.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - noisy_signal: 1D float32 array equal to clean_signal + noise.
            - noise: 1D float32 array containing the generated AWGN samples w[n].

    Raises:
        ValueError: If signal is empty.
    """
    if signal.size == 0:
        raise ValueError("Cannot add noise to an empty signal.")

    # Initialize independent NumPy random generator
    rng = np.random.default_rng(seed)
    n_samples = len(signal)

    # 1. Calculate clean signal power
    p_signal = float(np.mean(np.square(signal, dtype=np.float64)))

    # 2. Determine noise standard deviation (sigma)
    if target_snr_db is not None:
        if p_signal <= 1e-12:
            # Signal is silent; produce small baseline noise
            sigma = 0.01
        else:
            snr_linear = 10.0 ** (target_snr_db / 10.0)
            p_noise = p_signal / snr_linear
            sigma = np.sqrt(p_noise)
    elif noise_factor is not None:
        peak_amp = float(np.max(np.abs(signal))) if signal.size > 0 else 1.0
        if peak_amp < 1e-6:
            peak_amp = 1.0
        sigma = noise_factor * peak_amp
    else:
        # Default moderate noise level
        sigma = 0.05

    # 3. Generate zero-mean White Gaussian Noise
    noise = rng.normal(loc=0.0, scale=sigma, size=n_samples).astype(np.float32)

    # 4. Synthesize noisy signal without modifying the original input
    clean_float = signal.astype(np.float32, copy=True)
    noisy_signal = clean_float + noise

    return noisy_signal, noise


def create_noisy_signal_pair(
    clean_signal: np.ndarray,
    target_snr_db: float = 15.0,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Helper function to generate a reproducible clean/noisy signal pair with ground-truth metrics.

    Args:
        clean_signal: 1D numpy array of clean input samples.
        target_snr_db: Target SNR in decibels (default 15.0 dB).
        seed: Random seed for repeatability.

    Returns:
        dict containing:
            - 'clean': np.ndarray, copy of the clean input signal
            - 'noisy': np.ndarray, synthesized noisy signal
            - 'noise': np.ndarray, isolated noise component
            - 'target_snr_db': float, requested SNR in dB
            - 'realized_snr_db': float, empirically measured SNR in dB
            - 'signal_power': float, power of the clean signal
            - 'noise_power': float, power of the realized noise sequence
    """
    clean_copy = clean_signal.astype(np.float32, copy=True)
    noisy_signal, noise = add_white_gaussian_noise(
        clean_copy, target_snr_db=target_snr_db, seed=seed
    )

    p_sig = float(np.mean(np.square(clean_copy, dtype=np.float64)))
    p_noise = float(np.mean(np.square(noise, dtype=np.float64)))

    if p_noise > 1e-15 and p_sig > 1e-15:
        realized_snr = 10.0 * np.log10(p_sig / p_noise)
    else:
        realized_snr = float("inf") if p_noise <= 1e-15 else float("-inf")

    return {
        "clean": clean_copy,
        "noisy": noisy_signal,
        "noise": noise,
        "target_snr_db": float(target_snr_db),
        "realized_snr_db": float(realized_snr),
        "signal_power": p_sig,
        "noise_power": p_noise,
    }


def validate_filter_cutoffs(
    filter_type: str,
    cutoff: Union[float, Tuple[float, float], List[float]],
    fs: int,
) -> Tuple[str, Union[float, Tuple[float, float]]]:
    """
    Validate filter cutoff frequencies against the Nyquist limit (fs / 2).

    SNS Principles:
    - Nyquist-Shannon Sampling Theorem states that the maximum reconstructible frequency
      without aliasing is f_Nyquist = fs / 2.
    - Continuous-to-discrete digital filter transformations require cutoffs strictly in (0, f_Nyquist).
    - For a Band-Pass filter, both cutoffs must satisfy:
          0 < low_cutoff < high_cutoff < f_Nyquist

    Args:
        filter_type: "lowpass", "highpass", or "bandpass" (case-insensitive).
        cutoff: Single frequency in Hz for low/high-pass, or (low_hz, high_hz) for band-pass.
        fs: Sampling rate in Hz.

    Returns:
        Tuple[str, Union[float, Tuple[float, float]]]: Normalized (filter_type, validated_cutoff).

    Raises:
        ValueError: If cutoff exceeds Nyquist, is <= 0, or is invalid for the chosen filter type.
    """
    if fs <= 0:
        raise ValueError(f"Sampling frequency must be positive: {fs}")

    nyquist = fs / 2.0
    normalized_type = filter_type.strip().lower().replace("-", "").replace("_", "")

    if normalized_type in ("lowpass", "low", "lp"):
        normalized_type = "lowpass"
        if isinstance(cutoff, (tuple, list)):
            if len(cutoff) != 1:
                raise ValueError("Low-pass filter requires exactly one cutoff frequency.")
            c_val = float(cutoff[0])
        else:
            c_val = float(cutoff)

        if c_val <= 0.0:
            raise ValueError(
                f"Invalid low-pass cutoff frequency: {c_val:.1f} Hz. "
                f"Cutoff must be strictly positive (> 0 Hz)."
            )
        if c_val >= nyquist:
            raise ValueError(
                f"Invalid low-pass cutoff frequency: {c_val:.1f} Hz. "
                f"Cutoff must be strictly less than the Nyquist frequency ({nyquist:.1f} Hz for fs={fs} Hz)."
            )
        return normalized_type, c_val

    elif normalized_type in ("highpass", "high", "hp"):
        normalized_type = "highpass"
        if isinstance(cutoff, (tuple, list)):
            if len(cutoff) != 1:
                raise ValueError("High-pass filter requires exactly one cutoff frequency.")
            c_val = float(cutoff[0])
        else:
            c_val = float(cutoff)

        if c_val <= 0.0:
            raise ValueError(
                f"Invalid high-pass cutoff frequency: {c_val:.1f} Hz. "
                f"Cutoff must be strictly positive (> 0 Hz)."
            )
        if c_val >= nyquist:
            raise ValueError(
                f"Invalid high-pass cutoff frequency: {c_val:.1f} Hz. "
                f"Cutoff must be strictly less than the Nyquist frequency ({nyquist:.1f} Hz for fs={fs} Hz)."
            )
        return normalized_type, c_val

    elif normalized_type in ("bandpass", "band", "bp"):
        normalized_type = "bandpass"
        if not isinstance(cutoff, (tuple, list)) or len(cutoff) != 2:
            raise ValueError(
                "Band-pass filter requires a tuple or list of two frequencies: (low_cutoff, high_cutoff)."
            )

        low_val = float(cutoff[0])
        high_val = float(cutoff[1])

        if low_val <= 0.0:
            raise ValueError(
                f"Invalid band-pass lower cutoff: {low_val:.1f} Hz. Must be strictly positive (> 0 Hz)."
            )
        if high_val >= nyquist:
            raise ValueError(
                f"Invalid band-pass upper cutoff: {high_val:.1f} Hz. "
                f"Must be strictly less than the Nyquist frequency ({nyquist:.1f} Hz for fs={fs} Hz)."
            )
        if low_val >= high_val:
            raise ValueError(
                f"Invalid band-pass cutoff ordering: lower cutoff ({low_val:.1f} Hz) must be "
                f"strictly less than upper cutoff ({high_val:.1f} Hz)."
            )
        return normalized_type, (low_val, high_val)

    else:
        raise ValueError(
            f"Unsupported filter type '{filter_type}'. "
            "Supported types are: 'lowpass', 'highpass', 'bandpass'."
        )


def apply_lowpass_filter(
    signal: np.ndarray,
    fs: int,
    cutoff_hz: float,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a zero-phase Low-Pass Butterworth filter.

    SNS Characteristics:
    - Passband: Frequencies below cutoff_hz pass with maximally flat gain (~1.0 / 0 dB).
    - Stopband: High frequencies are attenuated at -20 * order dB/decade.
    - Zero-phase filtering: `scipy.signal.sosfiltfilt` applies forward and backward passes,
      canceling phase lag so signal components retain their exact temporal alignment.

    Args:
        signal: 1D numpy array of audio samples.
        fs: Sampling frequency in Hz.
        cutoff_hz: Cutoff frequency (-3 dB point) in Hz.
        order: Filter order (default 4).

    Returns:
        np.ndarray: Filtered 1D float32 array (new allocation).
    """
    _, validated_cutoff = validate_filter_cutoffs("lowpass", cutoff_hz, fs)
    return _apply_butterworth_sos(signal, fs, "lowpass", validated_cutoff, order)


def apply_highpass_filter(
    signal: np.ndarray,
    fs: int,
    cutoff_hz: float,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a zero-phase High-Pass Butterworth filter.

    SNS Characteristics:
    - Passband: Frequencies above cutoff_hz pass with maximally flat gain.
    - Stopband: Low frequencies and DC drift are attenuated.

    Args:
        signal: 1D numpy array of audio samples.
        fs: Sampling frequency in Hz.
        cutoff_hz: Cutoff frequency (-3 dB point) in Hz.
        order: Filter order (default 4).

    Returns:
        np.ndarray: Filtered 1D float32 array (new allocation).
    """
    _, validated_cutoff = validate_filter_cutoffs("highpass", cutoff_hz, fs)
    return _apply_butterworth_sos(signal, fs, "highpass", validated_cutoff, order)


def apply_bandpass_filter(
    signal: np.ndarray,
    fs: int,
    low_cutoff_hz: float,
    high_cutoff_hz: float,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a zero-phase Band-Pass Butterworth filter.

    SNS Characteristics:
    - Passband: Frequencies strictly between low_cutoff_hz and high_cutoff_hz pass.
    - Stopbands: Both low-frequency rumble and high-frequency hiss/noise are rejected.

    Args:
        signal: 1D numpy array of audio samples.
        fs: Sampling frequency in Hz.
        low_cutoff_hz: Lower passband boundary in Hz.
        high_cutoff_hz: Upper passband boundary in Hz.
        order: Filter order (default 4).

    Returns:
        np.ndarray: Filtered 1D float32 array (new allocation).
    """
    _, validated_cutoff = validate_filter_cutoffs(
        "bandpass", (low_cutoff_hz, high_cutoff_hz), fs
    )
    return _apply_butterworth_sos(signal, fs, "bandpass", validated_cutoff, order)


def apply_filter(
    signal: np.ndarray,
    fs: int,
    filter_type: str,
    cutoff: Union[float, Tuple[float, float], List[float]],
    order: int = 4,
) -> np.ndarray:
    """
    Universal filter dispatcher validating cutoffs and applying Butterworth filtering.

    Args:
        signal: 1D numpy array of input samples.
        fs: Sampling frequency in Hz.
        filter_type: "lowpass", "highpass", or "bandpass".
        cutoff: Single cutoff in Hz or (low, high) tuple for bandpass.
        order: Filter order (default 4).

    Returns:
        np.ndarray: Filtered 1D float32 array.
    """
    norm_type, val_cutoff = validate_filter_cutoffs(filter_type, cutoff, fs)
    return _apply_butterworth_sos(signal, fs, norm_type, val_cutoff, order)


def _apply_butterworth_sos(
    signal: np.ndarray,
    fs: int,
    btype: str,
    cutoff: Union[float, Tuple[float, float]],
    order: int = 4,
) -> np.ndarray:
    """
    Internal helper applying Butterworth filter via Second-Order Sections (SOS)
    and zero-phase forward-backward filtering (sosfiltfilt).
    """
    if signal.size == 0:
        raise ValueError("Cannot filter an empty signal.")

    # 1. Design Butterworth filter in Second-Order Sections (SOS) for numerical stability
    sos = scipy.signal.butter(
        N=order,
        Wn=cutoff,
        btype=btype,
        analog=False,
        output="sos",
        fs=fs,
    )

    # 2. Input array preservation: work on float64 copy for high computational precision
    sig_double = signal.astype(np.float64, copy=True)

    # 3. Handle short signal edge case for padlen
    padlen = 3 * (2 * order)
    if len(sig_double) <= padlen:
        # For signals shorter than default padding, adjust padlen safely
        padlen = max(1, len(sig_double) - 1)

    # 4. Zero-phase filtering: eliminates phase delay and distortion
    filtered_double = scipy.signal.sosfiltfilt(sos, sig_double, padlen=padlen)

    return filtered_double.astype(np.float32)


def calculate_signal_power(signal: np.ndarray) -> float:
    """
    Calculate the average power of a discrete-time signal.

    Formula:
        P = (1 / N) * sum_{n=0}^{N-1} x[n]^2

    Args:
        signal: 1D numpy array of signal values.

    Returns:
        float: Average signal power. Returns 0.0 for empty arrays.
    """
    if signal.size == 0:
        return 0.0
    return float(np.mean(np.square(signal, dtype=np.float64)))


def calculate_snr(
    clean_signal: np.ndarray,
    processed_signal: np.ndarray,
) -> float:
    """
    Calculate the objective Signal-to-Noise Ratio (SNR) in decibels (dB).

    SNS Principle:
    SNR quantitatively measures signal quality relative to unwanted noise or distortion:
        error[n] = processed_signal[n] - clean_signal[n]
        P_signal = (1 / N) * sum_{n=0}^{N-1} (clean_signal[n])^2
        P_error  = (1 / N) * sum_{n=0}^{N-1} (error[n])^2
        SNR_dB   = 10 * log10(P_signal / P_error)

    Honesty & Scientific Integrity:
    This calculation requires knowledge of the true clean signal x[n]. It is
    strictly valid when:
    1. A known synthetic signal was generated.
    2. Controlled noise was added to a known clean reference.
    It must NEVER be fabricated for arbitrary unreferenced real-world recordings.

    Args:
        clean_signal: 1D numpy array representing ground-truth clean audio x[n].
        processed_signal: 1D numpy array representing degraded or filtered audio y[n].

    Returns:
        float: Calculated SNR in decibels.
               Returns inf if processed is mathematically identical to clean.
               Returns -inf if clean signal has zero power.

    Raises:
        ValueError: If signals are empty or lengths do not match.
    """
    if clean_signal.size == 0 or processed_signal.size == 0:
        raise ValueError("Cannot calculate SNR of empty signals.")

    if len(clean_signal) != len(processed_signal):
        raise ValueError(
            f"Signal length mismatch for SNR calculation: "
            f"clean ({len(clean_signal)}) vs processed ({len(processed_signal)})."
        )

    clean_64 = clean_signal.astype(np.float64)
    proc_64 = processed_signal.astype(np.float64)

    p_signal = float(np.mean(np.square(clean_64)))
    p_error = float(np.mean(np.square(proc_64 - clean_64)))

    if p_signal <= 1e-15:
        return float("-inf")
    if p_error <= 1e-15:
        return float("inf")

    return float(10.0 * np.log10(p_signal / p_error))


def evaluate_filtering_performance(
    clean_signal: np.ndarray,
    noisy_signal: np.ndarray,
    filtered_signal: np.ndarray,
) -> Dict[str, Any]:
    """
    Evaluate the end-to-end performance of a noise reduction filter against clean reference:
        clean -> add noise -> noisy -> filter -> evaluate

    Calculates:
    - SNR before filtering (clean vs noisy)
    - SNR after filtering (clean vs filtered)
    - SNR improvement (delta in dB)
    - Power of signal, noise, and residual error

    Args:
        clean_signal: Known clean ground truth signal.
        noisy_signal: Degraded signal after noise addition.
        filtered_signal: Signal reconstructed by the digital filter.

    Returns:
        dict: Performance summary metrics.
    """
    snr_before = calculate_snr(clean_signal, noisy_signal)
    snr_after = calculate_snr(clean_signal, filtered_signal)
    improvement = snr_after - snr_before

    p_sig = calculate_signal_power(clean_signal)
    p_noise = calculate_signal_power(noisy_signal - clean_signal)
    p_residual = calculate_signal_power(filtered_signal - clean_signal)

    return {
        "snr_before_db": float(snr_before),
        "snr_after_db": float(snr_after),
        "snr_improvement_db": float(improvement),
        "signal_power": float(p_sig),
        "noise_power_before": float(p_noise),
        "residual_power_after": float(p_residual),
    }


def compute_filter_response(
    fs: int,
    filter_type: str,
    cutoff: Union[float, Tuple[float, float], List[float]],
    order: int = 4,
    worN: int = 1024,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the theoretical frequency response H(e^{j omega}) of the Butterworth filter.

    SNS Concepts:
    - Transfer function magnitude: |H(f)|
    - Magnitude in decibels: |H(f)|_dB = 20 * log10(|H(f)|)
    - Effective zero-phase response (sosfiltfilt):
      Because the signal is filtered forward and backward, the net frequency transfer function
      is H_eff(s) = H(s) * H(-s) = |H(s)|^2.
      In decibels: |H_eff(f)|_dB = 2 * |H(f)|_dB = 40 * log10(|H(f)|).
      The phase delay is identically zero: angle(H_eff) = 0 degrees across all frequencies.

    Args:
        fs: Sampling frequency in Hz.
        filter_type: "lowpass", "highpass", or "bandpass".
        cutoff: Single cutoff in Hz or (low, high) tuple.
        order: Filter order (default 4).
        worN: Number of frequency points to compute between 0 and Nyquist.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - freqs: 1D array of frequency points in Hz from 0 to fs/2.
            - mag_single_pass_db: Single-pass magnitude response in dB.
            - mag_effective_db: Net zero-phase magnitude response in dB (after sosfiltfilt).
    """
    norm_type, val_cutoff = validate_filter_cutoffs(filter_type, cutoff, fs)

    sos = scipy.signal.butter(
        N=order,
        Wn=val_cutoff,
        btype=norm_type,
        analog=False,
        output="sos",
        fs=fs,
    )

    freqs, h = scipy.signal.sosfreqz(sos, worN=worN, fs=fs)

    # Calculate magnitude in dB (clamp to -120 dB floor to avoid log(0))
    mag = np.abs(h)
    mag_clamped = np.maximum(mag, 1e-6)
    mag_single_db = 20.0 * np.log10(mag_clamped)
    mag_effective_db = 40.0 * np.log10(mag_clamped)

    return freqs, mag_single_db, mag_effective_db



