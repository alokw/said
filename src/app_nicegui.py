"""
Said - A simple text-to-speech app for audio engineers
Creates digital slate voiceovers for production workflows
"""
import os
import time
import warnings
from pathlib import Path
from nicegui import ui, app
import torch
import soundfile as sf
import numpy as np
from kokoro import KPipeline

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

# Initialize Kokoro pipeline (lazy loaded)
_pipeline = None

AVAILABLE_VOICES = [
    'af_heart', 'af_bella', 'af_nicole', 'af_sarah', 'af_sky',
    'am_michael', 'am_adam', 'am_eric',
    'bf_emma', 'bf_isabella',
    'bm_george', 'bm_lewis',
]

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

class Said:
    def __init__(self):
        self.output_dir = Path.home() / "Desktop" / "Said"
        self.output_dir.mkdir(exist_ok=True)
        self.current_voice = 'af_heart'
        self.current_speed = 1.0
        self.is_processing = False
        self.num_channels = 8
        self.include_descriptions = True
        self.channel_descriptions = {}
        self.export_mode = 'individual'  # 'individual', 'stereo_pairs', 'multichannel'
        self.target_dbtp = -1.0
        self.target_lufs = -18.0
        self.bit_depth = '16'  # '16' or '24'
        self.sample_rate = 48000  # 44100, 48000, or 96000
        self.progress_label = None
        self.spinner = None

    def show_progress(self, message: str):
        """Show progress overlay with spinner"""
        if self.spinner is None:
            with ui.dialog() as self.spinner, ui.card().classes('w-96 p-8 text-center'):
                ui.spinner(size='3rem').classes('mb-4')
                self.progress_label = ui.label(message).classes('text-lg')
                ui.label('This may take a moment...').classes('text-gray-500 mt-2')
        else:
            self.progress_label.text = message

        self.spinner.open()

    def hide_progress(self):
        """Hide progress overlay"""
        if self.spinner:
            self.spinner.close()
            self.spinner = None
            self.progress_label = None

    def init_pipeline(self):
        global _pipeline
        if _pipeline is None:
            # Enable MPS on Apple Silicon
            if torch.backends.mps.is_available():
                os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            _pipeline = KPipeline(lang_code='a')
        return _pipeline

    def resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio from original sample rate to target sample rate"""
        if orig_sr == target_sr:
            return audio

        if RESAMPLE_AVAILABLE:
            # Use scipy for high-quality resampling
            from scipy.signal import resample_poly
            # Calculate resampling ratio
            ratio = target_sr / orig_sr
            # Simplify the ratio to keep computation manageable
            from fractions import Fraction
            frac = Fraction(ratio).limit_denominator(1000)
            up = frac.numerator
            down = frac.denominator
            return resample_poly(audio, up, down)
        else:
            # Fallback: simple linear interpolation
            orig_len = len(audio)
            new_len = int(orig_len * target_sr / orig_sr)
            indices = np.linspace(0, orig_len - 1, new_len)
            return np.interp(indices, np.arange(orig_len), audio)

    def get_subtype(self):
        """Get soundfile subtype for bit depth"""
        return 'PCM_16' if self.bit_depth == '16' else 'PCM_24'

    def prepare_audio(self, audio: np.ndarray) -> np.ndarray:
        """Prepare audio: resample and normalize"""
        # Resample from Kokoro's 24kHz to target sample rate
        audio = self.resample_audio(audio, 24000, self.sample_rate)
        # Normalize
        audio = self.normalize_audio(audio, self.sample_rate)
        return audio

    def prepare_audio_multichannel(self, audio: np.ndarray) -> np.ndarray:
        """Prepare multichannel audio: resample and normalize"""
        # Resample from Kokoro's 24kHz to target sample rate
        if audio.ndim == 1:
            audio = self.resample_audio(audio, 24000, self.sample_rate)
        else:
            # Process each channel
            channels = []
            for ch in range(audio.shape[1]):
                channels.append(self.resample_audio(audio[:, ch], 24000, self.sample_rate))
            audio = np.column_stack(channels)
        # Normalize
        audio = self.normalize_audio(audio, self.sample_rate)
        return audio
        global _pipeline
        if _pipeline is None:
            ui.notify('Loading TTS model... (this may take a moment)', type='info')
            # Enable MPS on Apple Silicon
            if torch.backends.mps.is_available():
                os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            _pipeline = KPipeline(lang_code='a')
            ui.notify('Model loaded!', type='positive')
        return _pipeline

    def normalize_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Normalize audio to target dBTP peak and LUFS integrated"""
        # Peak normalization to target dBTP (dB True Peak)
        target_peak = 10 ** (self.target_dbtp / 20)
        peak = np.abs(audio).max()

        if peak > 0:
            audio = audio * (target_peak / peak)

        # LUFS normalization to target (if pyloudnorm is available)
        if LOUDNORM_AVAILABLE:
            try:
                meter = Meter(sample_rate)  # default: ITU-R BS.1770-4
                loudness = meter.integrated_loudness(audio)

                # Avoid division by zero or extreme gain
                if loudness > -100:  # Valid loudness measurement
                    gain = 10 ** ((self.target_lufs - loudness) / 20)
                    # Limit gain to reasonable range
                    gain = np.clip(gain, 0.1, 10)
                    audio = audio * gain

                    # Re-apply peak limit after LUFS gain
                    peak = np.abs(audio).max()
                    if peak > target_peak:
                        audio = audio * (target_peak / peak)
            except Exception:
                # Fall back to peak-only normalization on error
                pass

        # Clip any remaining overs (safety)
        audio = np.clip(audio, -1.0, 1.0)

        return audio
        global _pipeline
        if _pipeline is None:
            ui.notify('Loading TTS model... (this may take a moment)', type='info')
            # Enable MPS on Apple Silicon
            if torch.backends.mps.is_available():
                os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            _pipeline = KPipeline(lang_code='a')
            ui.notify('Model loaded!', type='positive')
        return _pipeline

    def generate_slate(self, filename: str):
        if self.is_processing:
            return

        self.is_processing = True
        self.show_progress('Initializing TTS model...')

        try:
            pipeline = self.init_pipeline()
            self.progress_label.text = 'Generating channel audio...'

            channel_audios = {}  # Store audio for each channel
            channel_durations = {}  # Store duration of each (at original 24kHz)

            # Generate audio for all channels first
            for i in range(1, self.num_channels + 1):
                self.progress_label.text = f'Generating channel {i}/{self.num_channels}...'

                # Build the text for this channel
                desc = self.channel_descriptions.get(i, DEFAULT_DESCRIPTIONS.get(i, ""))
                # Convert "vo" to "v.o." for proper pronunciation
                if desc:
                    desc = desc.replace('vo', 'v.o.').replace('VO', 'V.O.')

                if self.include_descriptions and desc:
                    text = f"{i}... {desc}."
                else:
                    text = f"{i}."

                # Generate audio for this channel
                generator = pipeline(text, voice=self.current_voice, speed=self.current_speed)
                for _, _, audio in generator:
                    channel_audios[i] = audio
                    channel_durations[i] = len(audio)
                    break

            self.progress_label.text = 'Processing and exporting...'

            # Export based on mode
            if self.export_mode == 'individual':
                # Individual mono files
                for i, audio in channel_audios.items():
                    # Prepare audio (resample + normalize)
                    audio = self.prepare_audio(audio)

                    desc = self.channel_descriptions.get(i, DEFAULT_DESCRIPTIONS.get(i, ""))
                    desc_clean = desc.replace(' ', '').replace('.', '').replace('v.o.', 'vo')
                    channel_num = f"{i:02d}"
                    safe_filename = f"ch{channel_num}_{desc_clean}.wav"
                    output_path = self.output_dir / safe_filename
                    sf.write(str(output_path), audio, self.sample_rate, subtype=self.get_subtype())

                ui.notify(f'Saved {len(channel_audios)} individual files', type='positive')

            elif self.export_mode == 'stereo_pairs':
                # Stereo pairs - separate files, timeline-aligned
                # All files have same total duration for timeline alignment

                # Calculate total duration for alignment (at original 24kHz)
                total_duration = sum(channel_durations.values())

                pair_count = 0
                left_offset = 0

                for i in range(1, self.num_channels + 1, 2):
                    left_audio = channel_audios.get(i, np.array([]))
                    right_audio = channel_audios.get(i + 1, np.array([]))

                    # Create left track: silence + left audio + silence
                    left_silence_before = np.zeros(left_offset)
                    left_silence_after = np.zeros(total_duration - left_offset - len(left_audio))
                    left_full = np.concatenate((left_silence_before, left_audio, left_silence_after))

                    # Right starts after left finishes
                    right_start = left_offset + len(left_audio)
                    right_silence_before = np.zeros(right_start)
                    right_silence_after = np.zeros(total_duration - right_start - len(right_audio))
                    right_full = np.concatenate((right_silence_before, right_audio, right_silence_after))

                    # Create stereo file
                    stereo = np.column_stack((left_full, right_full))

                    # Prepare audio (resample + normalize)
                    stereo = self.prepare_audio_multichannel(stereo)

                    # Build filename: chXX-YY_descXX_desYY
                    desc_left = self.channel_descriptions.get(i, DEFAULT_DESCRIPTIONS.get(i, ""))
                    desc_right = self.channel_descriptions.get(i + 1, DEFAULT_DESCRIPTIONS.get(i + 1, ""))
                    desc_left_clean = desc_left.replace(' ', '').replace('.', '').replace('v.o.', 'vo')
                    desc_right_clean = desc_right.replace(' ', '').replace('.', '').replace('v.o.', 'vo')
                    safe_filename = f"ch{i:02d}-{i+1:02d}_{desc_left_clean}_{desc_right_clean}.wav"
                    output_path = self.output_dir / safe_filename
                    sf.write(str(output_path), stereo, self.sample_rate, subtype=self.get_subtype())
                    pair_count += 1

                    # Update offsets for next pair
                    left_offset += len(left_audio) + len(right_audio)

                ui.notify(f'Saved {pair_count} stereo pairs (timeline-aligned, normalized)', type='positive')

            elif self.export_mode == 'multichannel':
                # Multichannel - sequential playback (ch1, then ch2, then ch3, etc.)
                # Each channel plays on its own track at its own time position
                # Calculate total duration (sum of all channel durations)
                total_duration = sum(channel_durations.values())

                multichannel_tracks = []
                current_offset = 0

                for i in range(1, self.num_channels + 1):
                    if i in channel_audios:
                        audio = channel_audios[i]
                        duration = len(audio)

                        # Create track with: silence + audio + silence
                        leading_silence = np.zeros(current_offset)
                        trailing_silence = np.zeros(total_duration - current_offset - duration)

                        full_track = np.concatenate((leading_silence, audio, trailing_silence))
                        multichannel_tracks.append(full_track)

                        current_offset += duration

                if multichannel_tracks:
                    # Stack as multichannel (columns = channels)
                    multichannel_array = np.column_stack(multichannel_tracks)

                    # Prepare audio (resample + normalize)
                    multichannel_array = self.prepare_audio_multichannel(multichannel_array)

                    # Build filename: chXX-ZZ_multichannel
                    safe_filename = f"ch01-{self.num_channels:02d}_multichannel.wav"
                    output_path = self.output_dir / safe_filename
                    sf.write(str(output_path), multichannel_array, self.sample_rate, subtype=self.get_subtype())
                    ui.notify(f'Saved multichannel file (sequential, {self.num_channels} channels)', type='positive')

        except Exception as e:
            ui.notify(f'Error: {str(e)}', type='negative')
        finally:
            self.is_processing = False
            self.hide_progress()

    def update_channel_list(self):
        """Rebuild the channel list when num_channels changes"""
        # Clear existing rows
        self.channel_container.clear()
        self.channel_descriptions = {}

        with self.channel_container:
            for i in range(1, self.num_channels + 1):
                default_desc = DEFAULT_DESCRIPTIONS.get(i, "")
                with ui.row().classes('w-full items-center gap-2 py-1'):
                    # Channel number
                    ui.label(str(i)).classes('w-12 text-center font-mono text-sm')

                    # Description input
                    input = ui.input(
                        placeholder='description...',
                        value=default_desc
                    ).classes('flex-grow').props('dense')

                    # Store reference
                    input.props(f'data-channel="{i}"')
                    # Bind changes to our descriptions dict
                    input.on('update:modelValue', lambda: self._update_desc(i, input.value))

    def _update_desc(self, channel: int, value: str):
        self.channel_descriptions[channel] = value

    def generate_custom(self, text: str, filename: str):
        if self.is_processing or not text.strip():
            return

        if not filename.strip():
            filename = "custom"

        self.is_processing = True
        self.show_progress('Initializing TTS model...')

        try:
            pipeline = self.init_pipeline()
            self.progress_label.text = 'Generating audio...'

            # Generate audio
            generator = pipeline(text, voice=self.current_voice, speed=self.current_speed)
            for _, _, audio in generator:
                self.progress_label.text = 'Processing audio...'
                # Prepare audio (resample + normalize)
                audio = self.prepare_audio(audio)

                # Save the file
                safe_filename = filename if filename.endswith('.wav') else f"{filename}.wav"
                output_path = self.output_dir / safe_filename
                sf.write(str(output_path), audio, self.sample_rate, subtype=self.get_subtype())

                ui.notify(f'Saved: {output_path.name}', type='positive')
                break

        except Exception as e:
            ui.notify(f'Error: {str(e)}', type='negative')
        finally:
            self.is_processing = False
            self.hide_progress()

    def open_custom_dialog(self):
        with ui.dialog() as dialog, ui.card().classes('w-96'):
            ui.label('Custom TTS').classes('text-xl font-bold mb-4')

            custom_text = ui.textarea(
                placeholder='Enter custom text...',
                value='1. mix left. 2. mix right.'
            ).classes('w-full h-32')

            custom_filename = ui.input(
                placeholder='Filename...',
                value='custom'
            ).classes('w-full mt-4')

            with ui.row().classes('w-full justify-end gap-2 mt-4'):
                ui.button('Cancel', on_click=dialog.close).props('flat')
                ui.button('Generate', on_click=lambda: (
                    self.generate_custom(custom_text.value, custom_filename.value),
                    dialog.close()
                )).props('flat bg-primary text-white')

        dialog.open()

    def build_ui(self):
        with ui.column().classes('w-full max-w-2xl mx-auto p-8 gap-6'):
            # Header
            ui.label('🎙️ Said').classes('text-4xl font-bold text-primary')
            ui.label('Channel identification slate generator').classes('text-gray-400')

            # Voice selection
            with ui.row().classes('w-full gap-4 items-center'):
                ui.label('Voice:').classes('font-medium w-16')
                ui.select(
                    AVAILABLE_VOICES,
                    value=self.current_voice,
                    on_change=lambda e: setattr(self, 'current_voice', e.value)
                ).classes('flex-grow')

            # Speed control
            with ui.row().classes('w-full gap-4 items-center'):
                ui.label('Speed:').classes('font-medium w-16')
                ui.slider(
                    min=0.5, max=2, step=0.1, value=1.0,
                    on_change=lambda e: setattr(self, 'current_speed', e.value)
                ).classes('flex-grow')
                ui.label().bind_text_from(self, 'current_speed', lambda v: f'{v:.1f}x')

            # Normalization settings
            ui.label('Normalization:').classes('font-medium')
            with ui.row().classes('w-full gap-6 items-center'):
                with ui.row().classes('gap-2 items-center'):
                    ui.label('Peak:')
                    ui.number(
                        value=-1.0,
                        min=-20, max=0, step=0.5,
                        on_change=lambda e: setattr(self, 'target_dbtp', e.value)
                    ).classes('w-20').props('dense')
                    ui.label('dBTP').classes('text-gray-400')

                with ui.row().classes('gap-2 items-center'):
                    ui.label('Loudness:')
                    ui.number(
                        value=-18.0,
                        min=-40, max=0, step=1,
                        on_change=lambda e: setattr(self, 'target_lufs', e.value)
                    ).classes('w-20').props('dense')
                    ui.label('LUFS').classes('text-gray-400')

            # Audio format settings
            ui.label('Audio Format:').classes('font-medium')
            with ui.row().classes('w-full gap-6 items-center'):
                with ui.row().classes('gap-2 items-center'):
                    ui.label('Bit Depth:')
                    ui.radio(
                        ['16-bit', '24-bit'],
                        value='16-bit',
                        on_change=lambda e: setattr(self, 'bit_depth', e.value.replace('-bit', ''))
                    ).props('inline dense')

                with ui.row().classes('gap-2 items-center'):
                    ui.label('Sample Rate:')
                    ui.radio(
                        ['44.1 kHz', '48 kHz', '96 kHz'],
                        value='48 kHz',
                        on_change=lambda e: setattr(self, 'sample_rate', int(float(e.value.split()[0]) * 1000))
                    ).props('inline dense')

            # Number of channels
            with ui.row().classes('w-full gap-4 items-center'):
                ui.label('Channels:').classes('font-medium w-16')
                channel_input = ui.number(
                    value=8,
                    min=1,
                    max=64,
                    step=1,
                    on_change=lambda e: self._on_channel_count_change(e.value)
                ).classes('w-24').props('dense')

            # Channel list container (scrollable)
            ui.label('Channel Descriptions:').classes('font-medium')

            with ui.card().classes('w-full'):
                self.channel_container = ui.column().classes('w-full max-h-64 overflow-y-auto')

            # Initialize channel list
            self.update_channel_list()

            # Output mode
            ui.label('Output Format:').classes('font-medium')
            with ui.row().classes('w-full gap-6'):
                ui.radio(
                    ['Channel IDs with descriptions', 'Channel IDs only'],
                    value='Channel IDs with descriptions',
                    on_change=lambda e: setattr(self, 'include_descriptions', e.value == 'Channel IDs with descriptions')
                ).props('inline')

            # Export mode
            ui.label('Channel Export:').classes('font-medium')
            ui.label('How audio files should be organized').classes('text-gray-500 text-sm mb-2')
            with ui.row().classes('w-full gap-4'):
                ui.radio(
                    ['Individual files', 'Stereo pairs', 'Multichannel'],
                    value='Individual files',
                    on_change=lambda e: self._on_export_mode_change(e.value)
                ).props('inline')

            # Generate buttons
            ui.button('Generate Slate', on_click=lambda: self.generate_slate('slate')) \
                .props('rounded size-lg').classes('w-full bg-primary text-white')

            ui.button('Generate Custom', on_click=self.open_custom_dialog) \
                .props('rounded').classes('w-full')

            # Output location info
            ui.label(f'Output: {self.output_dir}').classes('text-gray-500 text-sm')

    def _on_channel_count_change(self, value):
        self.num_channels = int(value)
        self.update_channel_list()

    def _on_export_mode_change(self, value):
        mode_map = {
            'Individual files': 'individual',
            'Stereo pairs': 'stereo_pairs',
            'Multichannel': 'multichannel'
        }
        self.export_mode = mode_map.get(value, 'individual')


def main():
    said_app = Said()

    # Build the UI
    said_app.build_ui()

    # Configure and run NiceGUI
    ui.run(
        title='Said',
        port=8080,
        dark=True,
        reload=False,
        window_size=(600, 850),
    )


if __name__ == '__main__':
    main()
