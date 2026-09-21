# AGENTS.md — Antigravity Instructions

## Mission
Build the Audio Signal Analyzer and Noise Reduction System Using FFT as a small, reliable academic Signals and Systems project.

## Rules

### Inspect first
Before major changes:
- inspect the workspace
- read `01_PROJECT_SPEC.md`
- check Python version
- check existing project files
- check whether a virtual environment exists

### Work incrementally
For every major stage:
1. implement
2. run tests or a direct verification
3. inspect the result
4. fix errors
5. continue

Never generate a huge application and assume it works.

### Keep code beginner-friendly
Prefer:
- clear names
- small functions
- simple modules
- comments around DSP operations

Avoid unnecessary frameworks and abstraction.

### Offline
No cloud APIs, web services, databases, authentication or runtime internet dependency.

### DSP correctness
- FFT: `rfft`/`rfftfreq` is preferred for real audio.
- Filters: Butterworth + SOS + `sosfiltfilt` for offline processing.
- Validate cutoff against Nyquist.
- Keep original, noisy and filtered arrays separate.
- Never invent SNR measurements.

### Microphone
Recording is optional. If microphone access fails, WAV upload must still work.

### GUI
Prefer PySide6. Keep the GUI simple, readable and demo-friendly.

### Dependencies
Use only dependencies required for the project.

### Testing
At minimum test:
- valid WAV loading
- invalid input
- FFT frequency detection using a synthetic sine wave
- noise generation
- all three filter types
- invalid cutoff values
- SNR with a known clean reference
- WAV saving
- app startup

### Documentation
Maintain:
- `README.md`
- `requirements.txt`
- useful docstrings/comments
- later `VIVA.md`, `DEMO_SCRIPT.md`, `ARCHITECTURE.md`

### Scope control
Do not add ML, LLM, ESP32, custom electronics or LINA integration before the core application is complete.

### No fake completion
Do not say a feature is complete unless it was actually tested. If something could not be tested, state exactly what could not be tested.

## Final Checklist
- [ ] App launches
- [ ] WAV loads
- [ ] waveform works
- [ ] FFT works
- [ ] noise works
- [ ] low-pass works
- [ ] high-pass works
- [ ] band-pass works
- [ ] cutoff validation works
- [ ] SNR is honest/valid
- [ ] filtered WAV saves
- [ ] playback failure is handled safely
- [ ] README complete
- [ ] tests pass or known limitations are documented
