import numpy as np
import librosa
from typing import List, Tuple

def audio_files_to_numpy(
    raw_audio: np.ndarray,
    sample_rate: int,
    frame_length: int,
    hop_length_frame: int,
    min_duration: float
) -> np.ndarray:
    """Convert audio files to numpy arrays."""
    
    # Ensure minimum duration
    min_samples = int(min_duration * sample_rate)
    if len(raw_audio) < min_samples:
        raw_audio = np.pad(raw_audio, (0, min_samples - len(raw_audio)))
    
    # Split signal into frames
    audio_framed = librosa.util.frame(
        raw_audio,
        frame_length=frame_length,
        hop_length=hop_length_frame
    ).T
    
    return audio_framed

def numpy_audio_to_matrix_spectrogram(
    audio_framed: np.ndarray,
    dim_square_spec: int,
    n_fft: int,
    hop_length_fft: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert numpy audio to spectrogram matrices."""
    
    m_amp_db = np.zeros((audio_framed.shape[0], dim_square_spec, dim_square_spec))
    m_pha = np.zeros((audio_framed.shape[0], dim_square_spec, dim_square_spec))

    for i in range(audio_framed.shape[0]):
        stft_data = librosa.stft(
            audio_framed[i],
            n_fft=n_fft,
            hop_length=hop_length_fft
        )
        
        amp_db = librosa.amplitude_to_db(np.abs(stft_data))
        pha = np.angle(stft_data)
        
        m_amp_db[i] = amp_db
        m_pha[i] = pha
    
    return m_amp_db, m_pha

def scaled_in(matrix_spec: np.ndarray) -> np.ndarray:
    """Scale input spectrogram."""
    matrix_spec = (matrix_spec + 46) / 50
    return matrix_spec

def inv_scaled_ou(matrix_spec: np.ndarray) -> np.ndarray:
    """Inverse scale output spectrogram."""
    matrix_spec = matrix_spec * 82 + 6
    return matrix_spec

def matrix_spectrogram_to_numpy_audio(
    m_amp_db: np.ndarray,
    m_pha: np.ndarray,
    frame_length: int,
    hop_length_fft: int
) -> List[np.ndarray]:
    """Convert spectrogram matrices back to numpy audio."""
    
    list_audio = []
    
    for i in range(m_amp_db.shape[0]):
        amp_db = m_amp_db[i]
        pha = m_pha[i]
        
        # Combine amplitude and phase
        amp = librosa.db_to_amplitude(amp_db)
        stft_matrix = amp * np.exp(1j * pha)
        
        # Inverse STFT
        audio = librosa.istft(
            stft_matrix,
            hop_length=hop_length_fft,
            length=frame_length
        )
        
        list_audio.append(audio)
    
    return list_audio 