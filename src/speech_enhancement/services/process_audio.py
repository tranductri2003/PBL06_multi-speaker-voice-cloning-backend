import os
import numpy as np
import soundfile as sf
import librosa
from .process_func import (
    audio_files_to_numpy,
    numpy_audio_to_matrix_spectrogram,
    scaled_in,
)

def preprocess_audio_for_prediction(audio):

    # audio_dir: str,
    # audio_input: list,
    # sample_rate: int = 8000,
    # min_duration: float = 1.0,
    # frame_length: int = 8064,
    # hop_length_frame: int = 8064,
    # n_fft: int = 255,
    # hop_length_fft: int = 63
    # Process audio files
    audio_frames = audio_files_to_numpy(raw_audio, sample_rate, frame_length, hop_length_frame, min_duration=1.0)
    
    # Spectrogram dimensions
    dim_square_spec = int(n_fft / 2) + 1

    # Amplitude and phase of spectrogram
    m_amp_db_audio, m_pha_audio = numpy_audio_to_matrix_spectrogram(
        audio_frames, dim_square_spec, n_fft, hop_length_fft)

    # Global scaling
    X_in = scaled_in(m_amp_db_audio)

    # Reshape for model input
    X_in = X_in.reshape(X_in.shape[0], X_in.shape[1], X_in.shape[2], 1)
    
    return X_in, m_amp_db_audio, m_pha_audio
