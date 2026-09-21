# Antigravity Prompt Sequence

Use these prompts one at a time. Do not paste all prompts together.

## Prompt 1 — Inspect Only

```text
We are building an academic Signals and Systems mini project called:

"Audio Signal Analyzer and Noise Reduction System Using FFT"

I have placed the project specification and AGENTS.md in this workspace.

Before writing code, inspect the workspace and read:
- AGENTS.md
- 01_PROJECT_SPEC.md
- 03_IMPLEMENTATION_PLAN.md

Then:
1. Inspect the current folder.
2. Check the operating system and Python version.
3. Check whether a virtual environment exists.
4. Check whether this is an empty or existing project.
5. Propose the smallest reliable implementation plan.
6. Do not add optional features.
7. Do not generate the full application yet.

After inspection, report:
- what you found
- what files you will create first
- exact command you expect to use to run the project

Do not implement the full application yet.
```

## Prompt 2 — Foundation

```text
Now implement Phase 1 and Phase 2 from 03_IMPLEMENTATION_PLAN.md.

Create the simple Python project structure and implement WAV audio I/O.

Requirements:
- WAV loading
- WAV saving
- stereo-to-mono analysis support
- metadata extraction
- safe error handling
- requirements.txt
- README.md

Do not build the GUI yet.

After coding:
1. install dependencies if needed
2. run import checks
3. run tests
4. fix errors
5. show the final file tree

Do not claim success without testing.
```

## Prompt 3 — Waveform + FFT

```text
Now implement Phase 3 and Phase 4.

Create beginner-readable functions for:
- time axis
- waveform
- RMS
- FFT
- frequency axis
- one-sided magnitude spectrum

Add tests using synthetic signals.

Test a known sine-wave frequency and verify that the FFT peak is near the expected frequency.

Do not build the GUI yet.

Run tests and fix all failures.
```

## Prompt 4 — Noise

```text
Now implement Phase 5.

Add controlled white Gaussian noise.

Requirements:
- clean signal remains unchanged
- noisy signal is stored separately
- configurable noise strength
- reproducible random seed for tests
- helper logic for known clean/noisy pairs

Add tests.

Run the tests and fix errors.
```

## Prompt 5 — Filters

```text
Now implement Phase 6.

Implement:
- low-pass Butterworth filter
- high-pass Butterworth filter
- band-pass Butterworth filter
- cutoff validation

Use scipy.signal with a numerically stable implementation.

Requirements:
- validate cutoff against Nyquist
- validate band-pass lower < upper
- preserve input array
- return a new filtered array
- use a suitable offline zero-phase approach

Add unit tests using synthetic signals with known low and high frequency components.

Run tests and fix errors.
```

## Prompt 6 — SNR

```text
Now implement Phase 7.

Implement mathematically correct SNR for situations where a clean reference exists.

For the generated demo:
clean -> add known noise -> noisy -> filter

Calculate:
- SNR before
- SNR after

Do not fabricate values.

Add tests with synthetic signals and known noise.
Run tests and fix errors.
```

## Prompt 7 — GUI

```text
Now implement Phase 8.

Build the desktop GUI using PySide6 unless there is a strong reason to use another framework.

The GUI must include:
1. Load WAV
2. Signal metadata
3. Original waveform
4. Original FFT
5. Add noise
6. Noisy waveform
7. Noisy FFT
8. Filter type selection
9. Cutoff controls
10. Apply Filter
11. Filtered waveform
12. Filtered FFT
13. SNR before/after when valid
14. Play Original
15. Play Noisy
16. Play Filtered
17. Save Filtered WAV
18. Status/error messages

Keep it simple and readable.

Launch the app and manually verify the main workflow. Fix runtime/GUI errors.
```

## Prompt 8 — End-to-End Audit

```text
The core application should now exist.

Review it against:
- AGENTS.md
- 01_PROJECT_SPEC.md
- 03_IMPLEMENTATION_PLAN.md

Test:
load WAV
-> waveform
-> FFT
-> add noise
-> noisy waveform + FFT
-> select filter
-> apply filter
-> filtered waveform + FFT
-> save output

Also test invalid input and invalid cutoff values.

Fix every issue you can find.

Do not add advanced features yet.
```

## Prompt 9 — Optional Features

```text
Only if the core project is already stable, add these one at a time:
1. peak frequency display
2. filter frequency response
3. reset button
4. optional microphone recording

After each feature, run tests and launch the app. Remove any optional feature that threatens reliability.

Do not add ML, LLMs, cloud APIs, databases, ESP32 or LINA integration.
```

## Prompt 10 — Documentation

```text
Create/update:
- README.md
- VIVA.md
- DEMO_SCRIPT.md
- ARCHITECTURE.md

README:
- overview
- features
- setup
- run commands
- usage
- troubleshooting
- limitations

VIVA:
- 20 likely questions with simple student-friendly answers
- focus on sampling, FFT, noise, filters and SNR

DEMO_SCRIPT:
- exact demo steps
- what to click
- what graph to show
- what to say

ARCHITECTURE:
- block diagram
- modules
- data flow
- DSP pipeline

Use the actual code in the workspace so the documentation matches the implementation.
```

## Prompt 11 — Final Audit

```text
Perform a complete final audit.

Check:
- Python imports
- dependencies
- unit tests
- GUI startup
- WAV loading
- waveform
- FFT
- noise generation
- low-pass
- high-pass
- band-pass
- cutoff validation
- SNR
- save output
- playback handling
- documentation

Run the application, not only static checks.

Fix every issue you can find.

At the end provide:
1. final file tree
2. exact run command
3. test results
4. known limitations
5. the five most important SNS concepts I need to understand for viva

Do not say complete unless the application was actually tested.
```

## Prompt 12 — Teach the Student

```text
I am the student who has to present this project, and I am a beginner in Signals and Systems.

Using the ACTUAL code in this workspace, teach me the project from zero in this order:

1. audio signal
2. sampling
3. discrete-time signal
4. time domain
5. frequency domain
6. DFT
7. FFT
8. noise
9. low-pass filter
10. high-pass filter
11. band-pass filter
12. cutoff frequency
13. Butterworth filter
14. SNR
15. how each concept maps to the actual Python files/functions

For each concept:
- explain simply
- show the relevant file/function
- explain why we use it
- give one viva answer

Do not assume I know electronics or DSP.
