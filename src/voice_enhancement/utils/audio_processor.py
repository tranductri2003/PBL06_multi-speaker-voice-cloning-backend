from typing import Tuple
import numpy as np
from ..configs.voice_enhancement_config import VoiceEnhancementConfig
from .audio_utils import (
    audio_files_to_numpy,
    numpy_audio_to_matrix_spectrogram,
    scaled_in,
)

class AudioProcessor:
    def __init__(self, config: VoiceEnhancementConfig = VoiceEnhancementConfig()):
        self.config = config

    def preprocess_audio(self, raw_audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Preprocess audio for model prediction"""
        # Convert audio to frames
        audio_frames = audio_files_to_numpy(
            raw_audio, 
            self.config.SAMPLE_RATE,
            self.config.FRAME_LENGTH,
            self.config.HOP_LENGTH_FRAME,
            self.config.MIN_DURATION
        )
        
        # Get spectrograms
        m_amp_db_audio, m_pha_audio = numpy_audio_to_matrix_spectrogram(
            audio_frames,
            self.config.DIM_SQUARE_SPEC,
            self.config.N_FFT,
            self.config.HOP_LENGTH_FFT
        )

        X_in = scaled_in(m_amp_db_audio)
        X_in = X_in.reshape(X_in.shape[0], X_in.shape[1], X_in.shape[2], 1)
        
        return X_in, m_amp_db_audio, m_pha_audio 