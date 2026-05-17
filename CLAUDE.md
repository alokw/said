# saID - Project Reference for Claude

## Project Overview

saID is a text-to-speech application for audio engineers to create digital slate voiceovers. It's a native PySide6 (Qt) desktop app that generates professional WAV files using the Kokoro TTS model.

## Architecture

**UI Framework:** PySide6 (Qt6) - native desktop UI, not web-based
**TTS Engine:** Kokoro (Python-first, Apache 2.0 licensed)
**Audio Processing:** soundfile, scipy.signal, pyloudnorm
**Packaging:** PyInstaller for standalone Mac/Windows executables

## Key Files

- `src/app.py` - Main PySide6 application (all UI + audio logic)
- `build/said.spec` - PyInstaller configuration
- `requirements.txt` - Python dependencies

## Current Features

- 30 voice options (US/UK accents, male/female) with emoji labels
- 1-64 channels with custom descriptions
- 3 export modes: individual mono, stereo pairs, multichannel WAV
- 2 sequencing modes: sequential (timeline-aligned) or concurrent
- Audio: 16/24-bit, 44.1/48/96 kHz, dBTP/LUFS normalization
- Custom TTS generation for arbitrary text
- Status bar with download/loading feedback

## UI Layout (PySide6)

```
┌─────────────────────────────────────────────────────┐
│  SAID                                    [Status]  │
├──────────────┬──────────────────────────────────────┤
│              │  Channel Descriptions                │
│  Sidebar     │  ┌────┬────────────────────┐          │
│  - Voice     │  │ Ch │ Description       │          │
│  - Channels  │  ├────┼────────────────────┤          │
│  - Export    │  │ 1  │ mix left           │          │
│  - Sequenc.  │  │ 2  │ mix right          │          │
│  - Content   │  └────┴────────────────────┘          │
│  - Bit Depth │                                       │
│  - Sample Rt │  [Generate Custom] [Generate Slates] │
│  - Normaliz. │                                       │
│  - Output    │                                       │
└──────────────┴──────────────────────────────────────┘
```

## Important Patterns

### Threaded Audio Generation

Audio processing happens in `AudioWorker` (QThread) to prevent UI freezing. Progress dialog updates via signals.

```python
class AudioWorker(QThread):
    progress = Signal(str)
    status = Signal(str)
    finished = Signal(bool, str)
    error = Signal(str)
```

### Pipeline Caching

The Kokoro pipeline is cached globally (`_pipeline`) to avoid reloading on every generation.

### Export Logic by Mode

- **Individual:** Loop through channels, write separate files. Sequential mode adds leading silence.
- **Stereo Pairs:** Pair (1,2), (3,4), etc. Sequential offsets in time, concurrent aligns at 0.
- **Multichannel:** Column stack all channels. Sequential offsets, concurrent aligns at 0.

### Audio Processing Chain

1. Generate at 24kHz (Kokoro native)
2. Resample to target sample rate (scipy.signal.resample_poly)
3. Normalize to target dBTP (peak)
4. Normalize to target LUFS (loudness) if pyloudnorm available
5. Write with PCM_16 or PCM_24 subtype

## Common Tasks

### Adding a New Setting

1. Add state variable in `saIDWindow.__init__`
2. Add UI widget in `setup_ui()` (use `create_section()` for consistency)
3. Add handler method (`on_setting_changed`)
4. Pass to worker in `generate_slate()` or `generate_custom()`

### Modifying Export Behavior

Edit `AudioWorker.run()` - the three export modes are handled sequentially after channel generation.

### Voice List

`AVAILABLE_VOICES` dict maps display names (with flags/emoji) to Kokoro voice codes:
- US female: af_heart, af_bella, af_nicole, etc.
- US male: am_michael, am_fenrir, etc.
- UK female: bf_emma, bf_isabella, etc.
- UK male: bm_george, bm_lewis, etc.

## Build Notes

- macOS: `./build.sh` creates `dist/saID.app`
- Windows: `build.bat` creates `dist\saID.exe`
- Bundle size is ~400-450 MB due to PyTorch
- Code signing recommended for public distribution

## Known Constraints

- Python 3.10+ required (Kokoro 0.9.4+)
- espeak-ng required for phonemization (install via brew/ separate installer)
- Mobile platforms not supported with current architecture
- First run downloads ~82 MB model files

## User Feedback

Users are audio engineers who expect:
- Professional, native UI (not web-based)
- Precise control over audio parameters
- Reliable output for DAW workflows
- Clear feedback during operations
