"""
saID - A simple text-to-speech app for audio and video engineers
Creates digital slate voiceovers for production workflows

Native PySide6 UI
"""
import os
import warnings
from pathlib import Path
from typing import Dict, Optional
import torch
import soundfile as sf
import numpy as np
from kokoro import KPipeline
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QSpinBox, QDoubleSpinBox, QSlider, QComboBox,
    QPushButton, QTableWidget, QTableWidgetItem,
    QGroupBox, QRadioButton, QButtonGroup, QProgressDialog,
    QMessageBox, QFrame, QSizePolicy, QScrollArea, QFileDialog
)
from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QFont, QPalette, QColor, QIcon

try:
    from pyloudnorm import Meter
    LOUDNORM_AVAILABLE = True
except ImportError:
    LOUDNORM_AVAILABLE = False

try:
    from scipy.signal import resample_poly
    RESAMPLE_AVAILABLE = True
except ImportError:
    RESAMPLE_AVAILABLE = False

# Suppress warnings from kokoro/torch
warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn.modules.rnn')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch.nn.utils.weight_norm')
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

AVAILABLE_VOICES = {
    '🇺🇸 🚺 Heart': 'af_heart',
    '🇺🇸 🚺 Bella': 'af_bella',
    '🇺🇸 🚺 Nicole': 'af_nicole',
    '🇺🇸 🚺 Aoede': 'af_aoede',
    '🇺🇸 🚺 Kore': 'af_kore',
    '🇺🇸 🚺 Sarah': 'af_sarah',
    '🇺🇸 🚺 Nova': 'af_nova',
    '🇺🇸 🚺 Sky': 'af_sky',
    '🇺🇸 🚺 Alloy': 'af_alloy',
    '🇺🇸 🚺 Jessica': 'af_jessica',
    '🇺🇸 🚺 River': 'af_river',
    '🇺🇸 🚹 Michael': 'am_michael',
    '🇺🇸 🚹 Fenrir': 'am_fenrir',
    '🇺🇸 🚹 Puck': 'am_puck',
    '🇺🇸 🚹 Echo': 'am_echo',
    '🇺🇸 🚹 Eric': 'am_eric',
    '🇺🇸 🚹 Liam': 'am_liam',
    '🇺🇸 🚹 Onyx': 'am_onyx',
    '🇺🇸 🚹 Santa': 'am_santa',
    '🇺🇸 🚹 Adam': 'am_adam',
    '🇬🇧 🚺 Emma': 'bf_emma',
    '🇬🇧 🚺 Isabella': 'bf_isabella',
    '🇬🇧 🚺 Alice': 'bf_alice',
    '🇬🇧 🚺 Lily': 'bf_lily',
    '🇬🇧 🚹 George': 'bm_george',
    '🇬🇧 🚹 Fable': 'bm_fable',
    '🇬🇧 🚹 Lewis': 'bm_lewis',
    '🇬🇧 🚹 Daniel': 'bm_daniel',
}

DEFAULT_DESCRIPTIONS = {
    1: "mix left",
    2: "mix right",
    3: "music left",
    4: "music right",
    5: "v.o. left",
    6: "v.o. right",
    7: "sfx left",
    8: "sfx right",
}


class WorkerSignals(QObject):
    progress = Signal(str)
    status = Signal(str)
    finished = Signal(bool, str)
    error = Signal(str)


class AudioWorker(QThread):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.signals = WorkerSignals()
        self.pipeline = None
        self.num_channels = 8
        self.channel_descriptions = {}
        self.include_descriptions = True
        self.export_mode = 'individual'
        self.sequencing_mode = 'sequential'  # sequential or concurrent
        self.target_dbtp = -1.0
        self.target_lufs = -18.0
        self.bit_depth = '16'
        self.sample_rate = 48000
        self.current_voice = 'af_heart'
        self.current_speed = 1.0
        self.output_dir = Path.home() / "Desktop" / "Said"
        self.output_dir.mkdir(exist_ok=True)
        # Custom generation mode
        self.custom_mode = False
        self.custom_text = ""
        self.custom_filename = "custom"

    def init_pipeline(self):
        global _pipeline
        if _pipeline is None:
            self.signals.status.emit("Loading TTS model (~82MB, first run only)...")
            if torch.backends.mps.is_available():
                os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            _pipeline = KPipeline(lang_code='a')
            self.signals.status.emit("Model ready! All voices available.")
        return _pipeline

    def resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return audio
        if RESAMPLE_AVAILABLE:
            from scipy.signal import resample_poly
            from fractions import Fraction
            ratio = target_sr / orig_sr
            frac = Fraction(ratio).limit_denominator(1000)
            return resample_poly(audio, frac.numerator, frac.denominator)
        else:
            orig_len = len(audio)
            new_len = int(orig_len * target_sr / orig_sr)
            indices = np.linspace(0, orig_len - 1, new_len)
            return np.interp(indices, np.arange(orig_len), audio)

    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        target_peak = 10 ** (self.target_dbtp / 20)
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio * (target_peak / peak)
        if LOUDNORM_AVAILABLE:
            try:
                meter = Meter(self.sample_rate)
                loudness = meter.integrated_loudness(audio)
                if loudness > -100:
                    gain = 10 ** ((self.target_lufs - loudness) / 20)
                    gain = np.clip(gain, 0.1, 10)
                    audio = audio * gain
                    peak = np.abs(audio).max()
                    if peak > target_peak:
                        audio = audio * (target_peak / peak)
            except Exception:
                pass
        return np.clip(audio, -1.0, 1.0)

    def prepare_audio(self, audio: np.ndarray) -> np.ndarray:
        audio = self.resample_audio(audio, 24000, self.sample_rate)
        audio = self.normalize_audio(audio)
        return audio

    def prepare_audio_multichannel(self, audio: np.ndarray) -> np.ndarray:
        if audio.ndim == 1:
            audio = self.resample_audio(audio, 24000, self.sample_rate)
        else:
            channels = []
            for ch in range(audio.shape[1]):
                channels.append(self.resample_audio(audio[:, ch], 24000, self.sample_rate))
            audio = np.column_stack(channels)
        audio = self.normalize_audio(audio)
        return audio

    def get_subtype(self):
        return 'PCM_16' if self.bit_depth == '16' else 'PCM_24'

    def run(self):
        try:
            self.signals.progress.emit("Initializing TTS model...")
            pipeline = self.init_pipeline()

            # Custom generation mode - single text to file
            if self.custom_mode:
                self.signals.progress.emit("Generating custom audio...")
                generator = pipeline(self.custom_text, voice=self.current_voice, speed=self.current_speed)
                for _, _, audio in generator:
                    audio = self.prepare_audio(audio)
                    safe_filename = self.custom_filename if self.custom_filename.endswith('.wav') else f"{self.custom_filename}.wav"
                    output_path = self.output_dir / safe_filename
                    sf.write(str(output_path), audio, self.sample_rate, subtype=self.get_subtype())
                    self.signals.finished.emit(True, f"Saved: {output_path.name}")
                    return

            channel_audios = {}
            channel_durations = {}

            for i in range(1, self.num_channels + 1):
                self.signals.progress.emit(f"Generating channel {i}/{self.num_channels}...")
                desc = self.channel_descriptions.get(i, DEFAULT_DESCRIPTIONS.get(i, ""))
                if desc:
                    desc = desc.replace('vo', 'v.o.').replace('VO', 'V.O.')

                if self.include_descriptions and desc:
                    text = f"{i}... {desc}."
                else:
                    text = f"{i}."

                generator = pipeline(text, voice=self.current_voice, speed=self.current_speed)
                for _, _, audio in generator:
                    channel_audios[i] = audio
                    channel_durations[i] = len(audio)
                    break

            self.signals.progress.emit("Processing and exporting...")

            if self.export_mode == 'individual':
                max_duration = max(channel_durations.values())

                if self.sequencing_mode == 'sequential':
                    # Add leading silence for timeline placement + trailing silence for loop sync
                    current_offset = 0
                    for i, audio in channel_audios.items():
                        leading_silence = np.zeros(current_offset)
                        trailing_silence = np.zeros(max_duration - len(audio))
                        full_audio = np.concatenate((leading_silence, audio, trailing_silence))
                        full_audio = self.prepare_audio(full_audio)
                        desc = self.channel_descriptions.get(i, DEFAULT_DESCRIPTIONS.get(i, ""))
                        desc_clean = desc.replace(' ', '').replace('.', '').replace('v.o.', 'vo')
                        safe_filename = f"ch{i:02d}_{desc_clean}.wav"
                        output_path = self.output_dir / safe_filename
                        sf.write(str(output_path), full_audio, self.sample_rate, subtype=self.get_subtype())
                        current_offset += len(audio)
                else:  # concurrent
                    for i, audio in channel_audios.items():
                        # Pad to max duration for loop sync
                        trailing_silence = np.zeros(max_duration - len(audio))
                        full_audio = np.concatenate((audio, trailing_silence))
                        full_audio = self.prepare_audio(full_audio)
                        desc = self.channel_descriptions.get(i, DEFAULT_DESCRIPTIONS.get(i, ""))
                        desc_clean = desc.replace(' ', '').replace('.', '').replace('v.o.', 'vo')
                        safe_filename = f"ch{i:02d}_{desc_clean}.wav"
                        output_path = self.output_dir / safe_filename
                        sf.write(str(output_path), full_audio, self.sample_rate, subtype=self.get_subtype())
                self.signals.finished.emit(True, f"Saved {len(channel_audios)} individual files")

            elif self.export_mode == 'stereo_pairs':
                if self.sequencing_mode == 'sequential':
                    total_duration = sum(channel_durations.values())
                    left_offset = 0

                    for i in range(1, self.num_channels + 1, 2):
                        left_audio = channel_audios.get(i, np.array([]))
                        right_audio = channel_audios.get(i + 1, np.array([]))

                        left_silence_before = np.zeros(left_offset)
                        left_silence_after = np.zeros(total_duration - left_offset - len(left_audio))
                        left_full = np.concatenate((left_silence_before, left_audio, left_silence_after))

                        right_start = left_offset + len(left_audio)
                        right_silence_before = np.zeros(right_start)
                        right_silence_after = np.zeros(total_duration - right_start - len(right_audio))
                        right_full = np.concatenate((right_silence_before, right_audio, right_silence_after))

                        stereo = np.column_stack((left_full, right_full))
                        stereo = self.prepare_audio_multichannel(stereo)

                        desc_left = self.channel_descriptions.get(i, DEFAULT_DESCRIPTIONS.get(i, ""))
                        desc_right = self.channel_descriptions.get(i + 1, DEFAULT_DESCRIPTIONS.get(i + 1, ""))
                        desc_left_clean = desc_left.replace(' ', '').replace('.', '').replace('v.o.', 'vo')
                        desc_right_clean = desc_right.replace(' ', '').replace('.', '').replace('v.o.', 'vo')
                        safe_filename = f"ch{i:02d}-{i+1:02d}_{desc_left_clean}_{desc_right_clean}.wav"
                        output_path = self.output_dir / safe_filename
                        sf.write(str(output_path), stereo, self.sample_rate, subtype=self.get_subtype())

                        left_offset += len(left_audio) + len(right_audio)
                else:  # concurrent
                    for i in range(1, self.num_channels + 1, 2):
                        left_audio = channel_audios.get(i, np.array([]))
                        right_audio = channel_audios.get(i + 1, np.array([]))

                        # Pad to match max duration
                        max_len = max(len(left_audio), len(right_audio))
                        left_padded = np.concatenate((left_audio, np.zeros(max_len - len(left_audio))))
                        right_padded = np.concatenate((right_audio, np.zeros(max_len - len(right_audio))))

                        stereo = np.column_stack((left_padded, right_padded))
                        stereo = self.prepare_audio_multichannel(stereo)

                        desc_left = self.channel_descriptions.get(i, DEFAULT_DESCRIPTIONS.get(i, ""))
                        desc_right = self.channel_descriptions.get(i + 1, DEFAULT_DESCRIPTIONS.get(i + 1, ""))
                        desc_left_clean = desc_left.replace(' ', '').replace('.', '').replace('v.o.', 'vo')
                        desc_right_clean = desc_right.replace(' ', '').replace('.', '').replace('v.o.', 'vo')
                        safe_filename = f"ch{i:02d}-{i+1:02d}_{desc_left_clean}_{desc_right_clean}.wav"
                        output_path = self.output_dir / safe_filename
                        sf.write(str(output_path), stereo, self.sample_rate, subtype=self.get_subtype())

                pair_count = (self.num_channels + 1) // 2
                self.signals.finished.emit(True, f"Saved {pair_count} stereo pairs")

            elif self.export_mode == 'multichannel':
                if self.sequencing_mode == 'sequential':
                    total_duration = sum(channel_durations.values())
                    multichannel_tracks = []
                    current_offset = 0

                    for i in range(1, self.num_channels + 1):
                        if i in channel_audios:
                            audio = channel_audios[i]
                            duration = len(audio)
                            leading_silence = np.zeros(current_offset)
                            trailing_silence = np.zeros(total_duration - current_offset - duration)
                            full_track = np.concatenate((leading_silence, audio, trailing_silence))
                            multichannel_tracks.append(full_track)
                            current_offset += duration
                else:  # concurrent
                    # Find max duration and pad all channels to match
                    max_duration = max(channel_durations.values())
                    multichannel_tracks = []

                    for i in range(1, self.num_channels + 1):
                        if i in channel_audios:
                            audio = channel_audios[i]
                            padding = np.zeros(max_duration - len(audio))
                            full_track = np.concatenate((audio, padding))
                            multichannel_tracks.append(full_track)

                if multichannel_tracks:
                    multichannel_array = np.column_stack(multichannel_tracks)
                    multichannel_array = self.prepare_audio_multichannel(multichannel_array)
                    safe_filename = f"ch01-{self.num_channels:02d}_multichannel.wav"
                    output_path = self.output_dir / safe_filename
                    sf.write(str(output_path), multichannel_array, self.sample_rate, subtype=self.get_subtype())
                    self.signals.finished.emit(True, f"Saved multichannel file ({self.num_channels} channels)")

        except Exception as e:
            self.signals.error.emit(f"Error: {str(e)}")


_pipeline = None


class SaidWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("saID")
        self.setMinimumSize(800, 550)
        self.resize(900, 650)

        # State
        self.output_dir = Path.home() / "Desktop" / "Said"
        self.output_dir.mkdir(exist_ok=True)
        self.num_channels = 8
        self.channel_descriptions = {}
        self.include_descriptions = True
        self.export_mode = 'multichannel'  # Changed default
        self.sequencing_mode = 'sequential'  # sequential or concurrent
        self.current_voice = 'af_heart'
        self.current_speed = 1.0
        self.target_dbtp = -1.0
        self.target_lufs = -18.0
        self.bit_depth = '16'
        self.sample_rate = 48000
        self.is_processing = False
        self.worker = None

        self.setup_ui()
        self.update_channel_list()

        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.setStyleSheet("""
            QStatusBar {
                background-color: #2d2d2d;
                color: #ccc;
                border-top: 1px solid #444;
                font-size: 12px;
            }
        """)
        self.status_bar.showMessage("Ready")

        # Menu bar with About
        menu_bar = self.menuBar()
        menu_bar.setStyleSheet("""
            QMenuBar {
                background-color: #2d2d2d;
                color: #ccc;
                border-bottom: 1px solid #444;
            }
            QMenuBar::item {
                padding: 4px 8px;
            }
            QMenuBar::item:selected {
                background-color: #3a82da;
            }
        """)

        help_menu = menu_bar.addMenu("Help")
        about_action = help_menu.addAction("About saID")
        about_action.triggered.connect(self.show_about)

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # === LEFT SIDEBAR (Settings) ===
        left_panel = QWidget()
        left_panel.setMinimumWidth(320)
        left_panel.setMaximumWidth(380)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(16, 16, 16, 16)
        left_layout.setSpacing(12)

        # Title
        title = QLabel("saID")
        title_font = QFont()
        title_font.setPointSize(28)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #2A82DA;")
        left_layout.addWidget(title)

        subtitle = QLabel("Slate / Audio ID Generator")
        subtitle.setStyleSheet("color: #888; font-size: 13px;")
        left_layout.addWidget(subtitle)

        left_layout.addSpacing(16)

        # Scroll area for settings
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(16)

        # Voice selection
        voice_group = self.create_section("Voice")
        voice_layout = QVBoxLayout()
        voice_layout.setSpacing(8)
        self.voice_combo = QComboBox()
        self.voice_combo.addItems(AVAILABLE_VOICES.keys())
        self.voice_combo.setCurrentIndex(0)
        self.voice_combo.currentTextChanged.connect(self.on_voice_changed)
        voice_layout.addWidget(self.voice_combo)
        voice_group.setLayout(voice_layout)
        scroll_layout.addWidget(voice_group)

        # Channels
        channel_group = self.create_section("Channels")
        channel_layout = QVBoxLayout()
        channel_layout.setSpacing(8)
        self.channel_spinbox = QSpinBox()
        self.channel_spinbox.setRange(1, 64)
        self.channel_spinbox.setValue(8)
        self.channel_spinbox.valueChanged.connect(self.on_channel_count_changed)
        channel_layout.addWidget(self.channel_spinbox)
        channel_group.setLayout(channel_layout)
        scroll_layout.addWidget(channel_group)

        # Export Mode
        export_group = self.create_section("Export Mode")
        export_layout = QVBoxLayout()
        export_layout.setSpacing(8)
        self.individual_radio = QRadioButton("Individual files")
        self.stereo_radio = QRadioButton("Stereo pairs")
        self.multi_radio = QRadioButton("Multichannel")
        self.multi_radio.setChecked(True)  # Changed default to multichannel
        self.individual_radio.toggled.connect(self.on_export_changed)
        self.stereo_radio.toggled.connect(self.on_export_changed)
        self.multi_radio.toggled.connect(self.on_export_changed)
        export_layout.addWidget(self.individual_radio)
        export_layout.addWidget(self.stereo_radio)
        export_layout.addWidget(self.multi_radio)
        export_group.setLayout(export_layout)
        scroll_layout.addWidget(export_group)

        # Sequencing
        sequencing_group = self.create_section("Sequencing")
        sequencing_layout = QVBoxLayout()
        sequencing_layout.setSpacing(8)
        self.seq_sequential_radio = QRadioButton("Sequential")
        self.seq_concurrent_radio = QRadioButton("Concurrent")
        self.seq_sequential_radio.setChecked(True)
        self.seq_sequential_radio.toggled.connect(self.on_sequencing_changed)
        self.seq_concurrent_radio.toggled.connect(self.on_sequencing_changed)
        sequencing_layout.addWidget(self.seq_sequential_radio)
        sequencing_layout.addWidget(self.seq_concurrent_radio)
        sequencing_group.setLayout(sequencing_layout)
        scroll_layout.addWidget(sequencing_group)

        # Output Format
        format_group = self.create_section("Content")
        format_layout = QVBoxLayout()
        format_layout.setSpacing(8)
        self.desc_radio = QRadioButton("ID + Description")
        self.id_radio = QRadioButton("ID only")
        self.desc_radio.setChecked(True)
        self.desc_radio.toggled.connect(self.on_format_changed)
        format_layout.addWidget(self.desc_radio)
        format_layout.addWidget(self.id_radio)
        format_group.setLayout(format_layout)
        scroll_layout.addWidget(format_group)

        # Bit Depth
        bit_group = self.create_section("Bit Depth")
        bit_layout = QVBoxLayout()
        bit_layout.setSpacing(8)
        self.bit_16_radio = QRadioButton("16-bit")
        self.bit_24_radio = QRadioButton("24-bit")
        self.bit_16_radio.setChecked(True)
        bit_layout.addWidget(self.bit_16_radio)
        bit_layout.addWidget(self.bit_24_radio)
        bit_group.setLayout(bit_layout)
        scroll_layout.addWidget(bit_group)

        # Sample Rate
        sr_group = self.create_section("Sample Rate")
        sr_layout = QVBoxLayout()
        sr_layout.setSpacing(8)
        self.sr_44_radio = QRadioButton("44.1 kHz")
        self.sr_48_radio = QRadioButton("48 kHz")
        self.sr_96_radio = QRadioButton("96 kHz")
        self.sr_48_radio.setChecked(True)
        sr_layout.addWidget(self.sr_44_radio)
        sr_layout.addWidget(self.sr_48_radio)
        sr_layout.addWidget(self.sr_96_radio)
        sr_group.setLayout(sr_layout)
        scroll_layout.addWidget(sr_group)

        # Normalization
        norm_group = self.create_section("Normalization")
        norm_layout = QVBoxLayout()
        norm_layout.setSpacing(10)

        # Peak
        peak_label = QLabel("Peak (dBTP)")
        peak_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        norm_layout.addWidget(peak_label)
        self.peak_spinbox = QSpinBox()
        self.peak_spinbox.setRange(-20, 0)
        self.peak_spinbox.setValue(-1)
        norm_layout.addWidget(self.peak_spinbox)

        # LUFS
        lufs_label = QLabel("Loudness (LUFS)")
        lufs_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        norm_layout.addWidget(lufs_label)
        self.lufs_spinbox = QSpinBox()
        self.lufs_spinbox.setRange(-40, 0)
        self.lufs_spinbox.setValue(-18)
        norm_layout.addWidget(self.lufs_spinbox)

        norm_group.setLayout(norm_layout)
        scroll_layout.addWidget(norm_group)

        scroll_layout.addStretch()

        scroll.setWidget(scroll_content)
        left_layout.addWidget(scroll)

        # Output location with browse button
        output_group = self.create_section("Output Folder")
        output_layout = QVBoxLayout()
        output_layout.setSpacing(8)

        output_path_layout = QHBoxLayout()
        self.output_label = QLabel(str(self.output_dir))
        self.output_label.setStyleSheet("color: #ccc; font-size: 12px;")
        self.output_label.setWordWrap(True)
        output_path_layout.addWidget(self.output_label)

        browse_btn = QPushButton("Browse...")
        browse_btn.setMaximumWidth(80)
        browse_btn.setStyleSheet("""
            QPushButton {
                background-color: #555;
                color: white;
                font-size: 12px;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QPushButton:hover {
                background-color: #666;
            }
        """)
        browse_btn.clicked.connect(self.browse_output_folder)
        output_path_layout.addWidget(browse_btn)

        output_layout.addLayout(output_path_layout)
        output_group.setLayout(output_layout)
        left_layout.addWidget(output_group)

        main_layout.addWidget(left_panel)

        # === RIGHT PANEL (Channel Table + Generate) ===
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(16, 16, 16, 16)
        right_layout.setSpacing(16)

        # Channel table header
        table_header = QLabel("Channel Descriptions")
        table_header_font = QFont()
        table_header_font.setPointSize(16)
        table_header_font.setBold(True)
        table_header.setFont(table_header_font)
        right_layout.addWidget(table_header)

        # Channel table
        self.channel_table = QTableWidget()
        self.channel_table.setColumnCount(2)
        self.channel_table.setHorizontalHeaderLabels(["Channel", "Description"])
        self.channel_table.verticalHeader().setVisible(False)
        self.channel_table.horizontalHeader().setStretchLastSection(False)
        self.channel_table.setColumnWidth(0, 80)
        self.channel_table.setColumnWidth(1, 180)  # Fixed width for description column
        self.channel_table.setAlternatingRowColors(True)
        self.channel_table.setShowGrid(True)
        right_layout.addWidget(self.channel_table)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)

        self.custom_button = QPushButton("Generate Custom String")
        self.custom_button.setMinimumHeight(40)
        self.custom_button.setMaximumHeight(40)
        self.custom_button.setMaximumWidth(180)
        self.custom_button.setStyleSheet("""
            QPushButton {
                background-color: #3a3a3a;
                color: #999;
                font-size: 12px;
                border-radius: 6px;
                padding: 8px 12px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
                color: #bbb;
            }
        """)
        self.custom_button.clicked.connect(self.open_custom_dialog)
        button_layout.addWidget(self.custom_button)

        self.generate_button = QPushButton("Generate Slates")
        self.generate_button.setMinimumHeight(40)
        self.generate_button.setMaximumHeight(40)
        self.generate_button.setStyleSheet("""
            QPushButton {
                background-color: #2A82DA;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 6px;
                padding: 8px 24px;
            }
            QPushButton:hover {
                background-color: #3A92EA;
            }
            QPushButton:pressed {
                background-color: #1A72CA;
            }
            QPushButton:disabled {
                background-color: #444;
                color: #888;
            }
        """)
        self.generate_button.clicked.connect(self.generate_slate)
        button_layout.addWidget(self.generate_button)

        right_layout.addLayout(button_layout)

        main_layout.addWidget(right_panel)

    def create_section(self, title: str) -> QGroupBox:
        """Create a consistent section/group box"""
        group = QGroupBox(title)
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 13px;
                border: 1px solid #444;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        return group

    def browse_output_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder",
            str(self.output_dir)
        )
        if folder:
            self.output_dir = Path(folder)
            self.output_label.setText(str(self.output_dir))

    def on_voice_changed(self, value):
        self.current_voice = AVAILABLE_VOICES.get(value, 'af_heart')
        self.status_bar.showMessage(f"Voice: {value} (ready - no download needed)")

    def on_speed_changed(self, value):
        self.current_speed = value / 10.0
        self.speed_label.setText(f"{self.current_speed:.1f}x")

    def on_channel_count_changed(self, value):
        self.num_channels = value
        self.update_channel_list()

    def on_format_changed(self):
        self.include_descriptions = self.desc_radio.isChecked()

    def on_export_changed(self):
        if self.individual_radio.isChecked():
            self.export_mode = 'individual'
        elif self.stereo_radio.isChecked():
            self.export_mode = 'stereo_pairs'
        else:
            self.export_mode = 'multichannel'

    def on_sequencing_changed(self):
        if self.seq_sequential_radio.isChecked():
            self.sequencing_mode = 'sequential'
        else:
            self.sequencing_mode = 'concurrent'

    def update_channel_list(self):
        self.channel_table.setRowCount(self.num_channels)
        self.channel_descriptions = {}

        for i in range(1, self.num_channels + 1):
            # Channel number
            num_item = QTableWidgetItem(str(i))
            num_item.setFlags(Qt.ItemIsEnabled)
            num_item.setTextAlignment(Qt.AlignCenter)
            self.channel_table.setItem(i - 1, 0, num_item)

            # Description
            desc = DEFAULT_DESCRIPTIONS.get(i, "")
            desc_item = QTableWidgetItem(desc)
            desc_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsEditable)
            self.channel_table.setItem(i - 1, 1, desc_item)
            self.channel_descriptions[i] = desc

    def gather_descriptions(self):
        """Gather descriptions from table"""
        for i in range(1, self.num_channels + 1):
            item = self.channel_table.item(i - 1, 1)
            if item:
                self.channel_descriptions[i] = item.text()

    def open_custom_dialog(self):
        from PySide6.QtWidgets import QDialog

        dialog = QDialog(self)
        dialog.setWindowTitle("Custom TTS")
        dialog.setMinimumWidth(500)
        layout = QVBoxLayout(dialog)

        layout.addWidget(QLabel("Enter custom text:"))
        text_input = QLineEdit()
        text_input.setText("1. mix left. 2. mix right.")
        layout.addWidget(text_input)

        layout.addWidget(QLabel("Filename:"))
        filename_input = QLineEdit()
        filename_input.setText("custom")
        layout.addWidget(filename_input)

        buttons = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)

        generate_btn = QPushButton("Generate")
        generate_btn.clicked.connect(lambda: self.generate_custom(text_input.text(), filename_input.text(), dialog))
        generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #2A82DA;
                color: white;
                padding: 8px 20px;
                border-radius: 4px;
            }
        """)

        buttons.addWidget(cancel_btn)
        buttons.addWidget(generate_btn)
        layout.addLayout(buttons)

        dialog.exec()

    def generate_custom(self, text, filename, dialog):
        if not text.strip():
            return
        if not filename.strip():
            filename = "custom"

        dialog.accept()
        self.is_processing = True
        self.generate_button.setEnabled(False)
        self.generate_button.setText("Generating...")

        # Show progress dialog
        self.progress_dialog = QProgressDialog("Initializing...", None, 0, 0, self)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setMinimumWidth(400)
        self.progress_dialog.setCancelButton(None)
        self.progress_dialog.setStyleSheet("""
            QProgressDialog {
                background-color: #333;
                color: white;
            }
            QLabel {
                color: white;
                font-size: 14px;
            }
        """)
        self.progress_dialog.show()

        # Start worker thread for custom generation
        self.worker = AudioWorker()
        self.worker.custom_mode = True
        self.worker.custom_text = text
        self.worker.custom_filename = filename
        self.worker.current_voice = self.current_voice
        self.worker.current_speed = self.current_speed
        self.worker.target_dbtp = float(self.peak_spinbox.value())
        self.worker.target_lufs = float(self.lufs_spinbox.value())
        self.worker.bit_depth = '16' if self.bit_16_radio.isChecked() else '24'

        if self.sr_44_radio.isChecked():
            self.worker.sample_rate = 44100
        elif self.sr_48_radio.isChecked():
            self.worker.sample_rate = 48000
        else:
            self.worker.sample_rate = 96000

        self.worker.output_dir = self.output_dir

        self.worker.signals.progress.connect(self.on_progress)
        self.worker.signals.status.connect(self.on_status)
        self.worker.signals.finished.connect(self.on_finished)
        self.worker.signals.error.connect(self.on_error)
        self.worker.start()

    def generate_slate(self):
        if self.is_processing:
            return

        self.gather_descriptions()
        self.is_processing = True
        self.generate_button.setEnabled(False)
        self.generate_button.setText("Generating...")

        # Show progress dialog
        self.progress_dialog = QProgressDialog("Initializing...", None, 0, 0, self)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setMinimumWidth(400)
        self.progress_dialog.setCancelButton(None)
        self.progress_dialog.setStyleSheet("""
            QProgressDialog {
                background-color: #333;
                color: white;
            }
            QLabel {
                color: white;
                font-size: 14px;
            }
        """)
        self.progress_dialog.show()

        # Start worker thread
        self.worker = AudioWorker()
        self.worker.num_channels = self.num_channels
        self.worker.channel_descriptions = self.channel_descriptions.copy()
        self.worker.include_descriptions = self.include_descriptions
        self.worker.export_mode = self.export_mode
        self.worker.sequencing_mode = self.sequencing_mode
        self.worker.target_dbtp = float(self.peak_spinbox.value())
        self.worker.target_lufs = float(self.lufs_spinbox.value())
        self.worker.bit_depth = '16' if self.bit_16_radio.isChecked() else '24'

        if self.sr_44_radio.isChecked():
            self.worker.sample_rate = 44100
        elif self.sr_48_radio.isChecked():
            self.worker.sample_rate = 48000
        else:
            self.worker.sample_rate = 96000

        self.worker.output_dir = self.output_dir

        self.worker.current_voice = self.current_voice
        self.worker.current_speed = self.current_speed

        self.worker.signals.progress.connect(self.on_progress)
        self.worker.signals.status.connect(self.on_status)
        self.worker.signals.finished.connect(self.on_finished)
        self.worker.signals.error.connect(self.on_error)
        self.worker.start()

    def on_progress(self, message):
        self.progress_dialog.setLabelText(message)

    def on_status(self, message):
        self.status_bar.showMessage(message)

    def on_finished(self, success, message):
        self.progress_dialog.close()
        self.is_processing = False
        self.generate_button.setEnabled(True)
        self.generate_button.setText("Generate Slates")
        self.status_bar.showMessage("Ready")
        QMessageBox.information(self, "Complete", message)

    def on_error(self, message):
        self.progress_dialog.close()
        self.is_processing = False
        self.generate_button.setEnabled(True)
        self.generate_button.setText("Generate Slates")
        self.status_bar.showMessage(f"Error: {message}")
        QMessageBox.critical(self, "Error", message)

    def show_about(self):
        about_text = """
        <h2>saID</h2>
        <p><b>Slate / Audio ID Generator</b></p>
        <p>A simple text-to-speech application for audio and video engineers<br>
        to create digital slate voiceovers.</p>
        <hr>
        <p><b>Powered by:</b></p>
        <ul>
            <li><b>Kokoro TTS</b> - Apache 2.0 licensed</li>
            <li><b>PySide6</b> - LGPL licensed</li>
            <li><b>PyTorch</b> - BSD licensed</li>
            <li><b>soundfile</b> - BSD licensed</li>
        </ul>
        <p><a href='https://github.com/hexgrad/kokoro'>Kokoro on GitHub</a></p>
        """
        QMessageBox.about(self, "About saID", about_text)


def set_dark_theme(app):
    """Apply dark theme for professional audio app feel"""
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(45, 45, 45))
    palette.setColor(QPalette.WindowText, QColor(220, 220, 220))
    palette.setColor(QPalette.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.AlternateBase, QColor(50, 50, 50))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, QColor(220, 220, 220))
    palette.setColor(QPalette.Button, QColor(55, 55, 55))
    palette.setColor(QPalette.ButtonText, QColor(220, 220, 220))
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(100, 150, 255))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)

    app.setPalette(palette)


def main():
    app = QApplication([])
    set_dark_theme(app)

    window = SaidWindow()
    window.show()
    app.exec()


if __name__ == '__main__':
    main()
