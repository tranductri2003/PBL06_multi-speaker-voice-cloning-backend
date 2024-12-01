from io import BytesIO
from pathlib import Path
import torch
import numpy as np
import soundfile as sf
import librosa

from speaker_verification.configs.audio_config import SpeakerEncoderAudioConfig
from core.utils.objects.utterance import Utterance
from core.utils.processors.audio_processor import AudioPreprocessor


def remove_silence(audio: np.ndarray, sr: int, threshold_db: float = -40, min_silence_duration: float = 0.1) -> np.ndarray:
    """
    Remove silence segments from audio
    
    Args:
        audio: Audio signal
        sr: Sample rate
        threshold_db: Threshold in decibels below reference to consider as silence
        min_silence_duration: Minimum silence duration in seconds
        
    Returns:
        Audio with silence removed
    """
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio)
    
    # Get non-silent intervals
    intervals = librosa.effects.split(
        audio,
        top_db=-threshold_db,
        frame_length=2048,
        hop_length=512
    )
    
    # Filter out short silence segments
    min_samples = int(min_silence_duration * sr)
    intervals = [
        [start, end] for start, end in intervals
        if end - start >= min_samples
    ]
    
    # Concatenate non-silent parts
    audio_clean = np.concatenate([audio[start:end] for start, end in intervals])
    
    return audio_clean


def preprocess_audio(file: BytesIO | str | Path) -> tuple[torch.Tensor, BytesIO, np.ndarray]:
    """Preprocess audio and return mel spectrogram tensor, cleaned audio, and mel visualization"""
    # Get raw audio data
    if isinstance(file, BytesIO):
        file.seek(0)
        audio_data, sr = sf.read(file)
    else:
        audio_data, sr = librosa.load(str(file), sr=SpeakerEncoderAudioConfig.SAMPLE_RATE)
    
    # Remove silence
    clean_audio = remove_silence(
        audio_data,
        sr=sr,
        threshold_db=-40,
        min_silence_duration=0.1
    )
    
    # Convert cleaned audio to BytesIO
    clean_audio_io = BytesIO()
    sf.write(clean_audio_io, clean_audio, sr, format='WAV')
    clean_audio_io.seek(0)
    
    # Create utterance from cleaned audio
    clean_uttn = Utterance(
        raw_file=clean_audio_io,
        processor=AudioPreprocessor(config=SpeakerEncoderAudioConfig)
    )
    
    # Get mel spectrogram for model input
    mel_spectrogram = torch.tensor(
        np.array(
            [clean_uttn.random_mel_in_db(num_frames=SpeakerEncoderAudioConfig.NUM_FRAMES)]
        )
    ).transpose(1, 2)
    
    # Get full mel spectrogram for visualization
    mel_viz = clean_uttn.mel_in_db()
    
    clean_audio_io.seek(0)
    return mel_spectrogram, clean_audio_io, mel_viz
