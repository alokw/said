# 🎙️ saID

A simple text-to-speech application for audio engineers to create digital slate voiceovers.

**saID** — play on words for both "say ID" (identification) and traditional slate marking.

**Features:**
- Local-only operation (no internet required after first run)
- High-quality Kokoro TTS model (Apache 2.0 licensed)
- 30 voice options (US/UK accents, male/female)
- Multiple export formats: individual mono files, stereo pairs, or multichannel WAV
- Sequential or concurrent channel timing
- Professional audio features: normalization (dBTP/LUFS), sample rate conversion (44.1/48/96 kHz), bit depth selection (16/24-bit)
- Native PySide6 UI for Mac and Windows

## Platform Support

| Platform | Status | Package Size | Notes |
|----------|--------|--------------|-------|
| macOS (Intel + Apple Silicon) | ✅ Native build | ~400 MB | MPS GPU acceleration on M1/M2/M3/M4 |
| Windows (x64) | ✅ Native build | ~450 MB | Requires espeak-ng installation |
| Linux | ✅ Works (no official build) | ~400 MB | Community builds only |

## Quick Start (Testing Locally)

### Prerequisites

**Python Version:** 3.10 - 3.12 required (3.11 recommended)

**macOS:**
```bash
brew install python@3.11
brew install espeak-ng
```

**Windows:**
1. Install Python 3.10+ from [python.org](https://www.python.org/downloads/)
2. Download and install [espeak-ng](https://github.com/espeak-ng/espeak-ng/releases)

### Setup

```bash
cd said
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
pip install --upgrade pip
pip install -r requirements.txt
```

### Running

```bash
python src/app.py
```

**First Run:** The app will download the Kokoro TTS model (~82 MB) when you first generate audio. This happens automatically and only once. Download progress appears in the status bar at the bottom of the window.

**Subsequent Launches:** The app loads the cached model and is ready to generate immediately.

**Voice Selection:** All 30 voices are included with the model download - switching voices is instant and requires no additional downloads.

## Usage Guide

### Interface Overview

- **Left Sidebar:** Voice selection, channels, export mode, sequencing, content format, audio settings, normalization, output folder
- **Right Panel:** Channel descriptions table with generate buttons

### Settings

| Setting | Options | Description |
|---------|---------|-------------|
| **Voice** | 30 voices (US/UK, M/F) | Select TTS voice |
| **Channels** | 1-64 | Number of channels to generate |
| **Export Mode** | Individual / Stereo Pairs / Multichannel | Output file format |
| **Sequencing** | Sequential / Concurrent | Channel timing |
| **Content** | ID + Description / ID only | What to speak |
| **Bit Depth** | 16-bit / 24-bit | WAV bit depth |
| **Sample Rate** | 44.1 / 48 / 96 kHz | Output sample rate |
| **Peak (dBTP)** | -20 to 0 dB | True peak normalization |
| **Loudness (LUFS)** | -40 to 0 LUFS | LUFS normalization |

### Export Modes Explained

**Individual Files:** One mono WAV per channel
- Sequential: Each file has leading silence for timeline placement
- Concurrent: All files start at 0:00

**Stereo Pairs:** Pairs channels (1-2, 3-4, etc.) into stereo files
- Sequential: Channels play in sequence (L→R, then next pair)
- Concurrent: L+R play simultaneously

**Multichannel:** All channels in one WAV file
- Sequential: Channels play one after another
- Concurrent: All channels start together

### Custom Generation

Click "Generate Custom String" to create arbitrary TTS audio with current settings.

## Building for Distribution

### macOS

```bash
./build.sh
# Output: dist/saID.app (~400 MB)
```

### Windows

```cmd
build.bat
# Output: dist\saID.exe (~450 MB)
```

## Project Structure

```
said/
├── src/
│   └── app.py              # PySide6 application
├── build/
│   └── said.spec           # PyInstaller config
├── assets/                 # Icons
├── requirements.txt        # Dependencies
├── build.sh                # macOS build script
└── build.bat               # Windows build script
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| kokoro | >=0.9.4 | TTS model and inference |
| PySide6 | latest | Native Qt GUI |
| torch | latest | PyTorch for model runtime |
| soundfile | latest | WAV file I/O |
| pyloudnorm | latest | LUFS loudness normalization |
| scipy | latest | Sample rate conversion |
| numpy | latest | Audio processing |

## License

This project uses the Kokoro TTS model (Apache 2.0 licensed).

## Resources

- [Kokoro GitHub](https://github.com/hexgrad/kokoro)
- [PySide6 Documentation](https://pyside6.readthedocs.io/)
- [PyInstaller Docs](https://pyinstaller.org/)
