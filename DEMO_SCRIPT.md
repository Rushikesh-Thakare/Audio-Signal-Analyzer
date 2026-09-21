# Academic Demonstration Script

**Project Title:** Audio Signal Analyzer and Noise Reduction System Using FFT  
**Course:** Signals and Systems (SNS) Mini Project  
**Presenter:** Student Presenter  
**Duration:** ~5 to 7 minutes  

---

## Preparation Before Demo

1. Ensure the Python virtual environment is active:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
2. Verify that synthetic sample files are present in `sample_audio/` (run `python sample_audio/generate_samples.py` if missing).
3. Connect headphones or computer speakers and check volume.
4. Launch the application:
   ```powershell
   python app.py
   ```

---

## Demonstration Steps

### Step 1: Introduction & System Overview
- **What to click:** None (app window is open on screen).
- **What to show:** The clean graphical interface with control panels on the left, tabbed visualization canvases on the right, and the initial status bar showing `"Ready. Please load a WAV file to begin."`.
- **What to say:**
  > *"Good morning respected evaluators. Today I am demonstrating our Signals and Systems mini project: an **Audio Signal Analyzer and Noise Reduction System Using FFT**.*  
  > *This application bridges theoretical concepts—such as discrete-time sampling, Fourier analysis, Additive White Gaussian Noise, and IIR Butterworth filtering—into an interactive visual and audible tool.*  
  > *Let's begin by loading a test audio file."*

---

### Step 2: Loading the Audio File & Inspecting Metadata
- **What to click:** Click **"Load WAV File"** button at the top left. Select `sample_audio/mixed_tone_300hz_3500hz.wav`.
- **What to show:**
  - The **Signal Metadata** card on the left panel:
    - **Sample Rate ($f_s$):** `16,000 Hz`
    - **Duration:** `2.00 s`
    - **Samples ($N$):** `32,000`
    - **Channels:** `Mono (1 ch)`
    - **RMS Energy:** `~0.499`
    - **Peak Freq (Orig):** `300.0 Hz`
    - **Nyquist Limit:** `8,000.0 Hz`
  - In the **Overview Tab**, the top row shows the clean time-domain waveform and its frequency spectrum.
- **What to say:**
  > *"I have loaded a synthetic audio signal sampled at $16\text{ kHz}$. It consists of two deliberate frequency components: a desired low-frequency $300\text{ Hz}$ tone and an unwanted high-frequency interference tone at $3500\text{ Hz}$.*  
  > *Notice the metadata panel: it shows an RMS energy of $0.5$ and automatically detects the dominant peak at $300\text{ Hz}$, highlighted by the red dashed line on our magnitude spectrum.*  
  > *Notice also that our frequency spectrum only extends up to $8\text{ kHz}$—exactly the Nyquist limit $f_s / 2$."*

---

### Step 3: Audio Playback of Original Signal
- **What to click:** In the **Playback Controls** card, click **"Play Original"**.
- **What to show:** Audio plays through speakers.
- **What to say:**
  > *"When we listen to the original file, we clearly hear the low pitch combined with a piercing high-pitched whine from the $3500\text{ Hz}$ interference."*

---

### Step 4: Injecting Controlled AWGN Noise
- **What to click:**
  - In the **Noise Generator** card, verify the **Target SNR (dB)** is set to `10.0`.
  - Click the **"Add Noise"** button.
- **What to show:**
  - The middle row of the **Overview Tab** updates with the **Noisy Waveform** and **Noisy FFT Spectrum**.
  - Notice the time-domain waveform now exhibits erratic random fluctuations.
  - In the FFT spectrum, a wide, flat noise floor appears uniformly across all frequencies from $0\text{ Hz}$ to $8\text{ kHz}$.
  - The **SNR Evaluation** card updates:
    - **SNR Before:** `10.0 dB` (matches our target setting exactly).
    - **SNR After:** `N/A` (since filter hasn't been applied yet).
- **What to say:**
  > *"Now we simulate realistic signal degradation by adding Additive White Gaussian Noise (AWGN) with a target SNR of $10\text{ dB}$.*  
  > *In the time domain, the clean sine waves are corrupted by random fluctuations.*  
  > *In the frequency domain, notice the elevated, flat noise floor spanning all frequencies. This illustrates why it is called 'White' noise—it has uniform power spectral density across the entire spectrum.*  
  > *Our measured SNR Before is confirmed at $10.0\text{ dB}$ using the ground-truth clean signal power divided by noise power."*
- **What to click:** Click **"Play Noisy"** to hear the audio with background hiss.

---

### Step 5: Filter Configuration & Frequency Response Inspection
- **What to click:**
  - In the **Digital Butterworth Filter** panel:
    - Set **Filter Type**: `Low-pass`
    - Set **Cutoff Freq (Hz)**: `1000.0`
  - In the right tabbed view, switch to the **"Filter Response"** tab.
- **What to show:**
  - The **Bode Magnitude Plot (dB)** and **Phase Plot (Degrees)**.
  - Show the $-3\text{ dB}$ horizontal dotted line intersecting the magnitude curve at exactly $1000\text{ Hz}$.
  - Point out that at $300\text{ Hz}$ (desired signal), attenuation is $0\text{ dB}$ (unity gain, $100\%$ pass).
  - Point out that at $3500\text{ Hz}$ (interference tone), attenuation is approximately $-45\text{ dB}$ (substantial suppression).
- **What to say:**
  > *"Before filtering, we inspect our filter's frequency response in the 'Filter Response' tab.*  
  > *We configured a 4th-order Low-Pass Butterworth filter with a cutoff of $1000\text{ Hz}$.*  
  > *Because Butterworth filters are maximally flat, there are no passband ripples. At $1000\text{ Hz}$, the magnitude drops by exactly $-3\text{ dB}$ (half power).*  
  > *At our unwanted tone frequency of $3500\text{ Hz}$, the filter provides over $40\text{ dB}$ of attenuation."*

---

### Step 6: Applying the Filter & Zero-Phase Analysis
- **What to click:**
  - Switch back to the **"Pipeline Overview"** tab.
  - Click the **"Apply Filter"** button.
- **What to show:**
  - The bottom row plots **Filtered Waveform** and **Filtered FFT Spectrum**.
  - In the filtered waveform, the smooth sinusoidal shape is restored.
  - In the filtered FFT, frequencies above $1000\text{ Hz}$—including the $3500\text{ Hz}$ spike and the high-frequency white noise—are eliminated.
  - In the **SNR Evaluation** card:
    - **SNR After:** shows an increase to `~13.5 dB` to `~14.5 dB`.
    - **SNR Improvement:** shows a positive gain of `+3.5 dB` to `+4.5 dB`.
- **What to say:**
  > *"Now we apply the filter. Our backend executes a zero-phase forward-backward Butterworth filter using Second-Order Sections (`sosfiltfilt`).*  
  > *Zero-phase processing completely cancels any phase delay, preserving time alignment.*  
  > *Looking at the filtered FFT, the $3500\text{ Hz}$ tone has vanished, and the noise floor above $1000\text{ Hz}$ has dropped dramatically.*  
  > *Quantitatively, our SNR improved from $10.0\text{ dB}$ to over $14\text{ dB}$, confirming an objective noise reduction."*

---

### Step 7: Listening to the Filtered Output
- **What to click:** In the **Playback Controls** card, click **"Play Filtered"**.
- **What to show:** Clean audio plays without the high-frequency whistle and with significantly reduced background hiss.
- **What to say:**
  > *"Audibly, the piercing $3500\text{ Hz}$ interference is completely gone, and the background hiss is muted, leaving only the pure low $300\text{ Hz}$ tone."*

---

### Step 8: Saving Output & Resetting State
- **What to click:**
  - Click **"Save Filtered WAV"**.
  - In the file dialog, save as `output/demo_filtered_output.wav`.
  - Notice the status bar confirms: `"Saved filtered audio to ... successfully."`
  - Click **"Reset All"** button and confirm the dialog.
- **What to show:**
  - All plots clear to blank grid lines.
  - All playback and filter buttons safely return to disabled states.
  - Metadata labels reset to default (`File: None`, `SNR: N/A`).
  - Status bar confirms `"Reset complete. System ready."`.
- **What to say:**
  > *"The filtered result can be saved back to disk with automatic amplitude clipping protection.*  
  > *Finally, the 'Reset All' button clears the pipeline and safely resets memory for the next experiment.*  
  > *This concludes our demonstration. We are now ready for questions."*
