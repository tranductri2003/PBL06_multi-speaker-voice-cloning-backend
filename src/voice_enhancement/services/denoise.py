from io import BytesIO
import numpy as np
import soundfile as sf
import librosa
from typing import Tuple
from dataclasses import dataclass
from core.utils.model import TensorFlowModel

from ..configs.voice_enhancement_config import VoiceEnhancementConfig
from ..utils.audio_processor import AudioProcessor
from ..utils.audio_utils import inv_scaled_ou, matrix_spectrogram_to_numpy_audio
from .visualization import SpectrogramVisualizer

@dataclass
class DenoiseResult:
    """Data class for storing denoising results"""
    output_audio: BytesIO
    spectrogram_voice_noise: bytes
    spectrogram_predicted_noise: bytes
    spectrogram_predicted_voice: bytes
    wave_voice_noise: bytes
    wave_predicted_noise: bytes
    wave_predicted_voice: bytes

class DenoiseService:
    def __init__(
        self,
        config: VoiceEnhancementConfig = VoiceEnhancementConfig(),
        processor: AudioProcessor = None,
        visualizer: SpectrogramVisualizer = None
    ):
        self.config = config
        self.processor = processor or AudioProcessor(config)
        self.visualizer = visualizer or SpectrogramVisualizer(config)
    
    def process_audio(
        self, 
        audio_bytes: bytes, 
        model: TensorFlowModel
    ) -> DenoiseResult:
        """Process audio for noise reduction"""
        # Load audio
        raw_audio, _ = librosa.load(BytesIO(audio_bytes), sr=self.config.SAMPLE_RATE)
        
        # Preprocess audio
        X_in, m_amp_db_audio, m_pha_audio = self.processor.preprocess_audio(raw_audio)
        
        # Make predictions
        X_pred = model.predict(X_in)
        inv_sca_X_pred = inv_scaled_ou(X_pred)
        X_denoise = m_amp_db_audio - inv_sca_X_pred[:, :, :, 0]

        # Reconstruct audio
        original_audio = self._reconstruct_audio(m_amp_db_audio, m_pha_audio)
        noised_audio = self._reconstruct_audio(inv_sca_X_pred[:, :, :, 0], m_pha_audio)
        denoised_audio = self._reconstruct_audio(X_denoise, m_pha_audio)
        output_audio = self._prepare_output_audio(denoised_audio)
        
        # Generate visualizations
        visualizations = self.visualizer.generate_visualizations(
            m_amp_db_audio[0],
            inv_sca_X_pred[0, :, :, 0],
            X_denoise[0],
            original_audio,
            noised_audio,
            denoised_audio
        )
        
        return DenoiseResult(
            output_audio=output_audio,
            spectrogram_voice_noise=visualizations[0],
            spectrogram_predicted_noise=visualizations[1],
            spectrogram_predicted_voice=visualizations[2],
            wave_voice_noise=visualizations[3],
            wave_predicted_noise=visualizations[4],
            wave_predicted_voice=visualizations[5]
        )
    
    def _reconstruct_audio(self, X_denoise: np.ndarray, m_pha_audio: np.ndarray) -> np.ndarray:
        """Reconstruct audio from spectrograms"""
        audio_denoise_recons = matrix_spectrogram_to_numpy_audio(
            X_denoise, 
            m_pha_audio, 
            self.config.FRAME_LENGTH, 
            self.config.HOP_LENGTH_FFT
        )
        return np.concatenate(audio_denoise_recons, axis=0)
    
    def _prepare_output_audio(self, denoised_audio: np.ndarray) -> BytesIO:
        """Prepare audio for output"""
        output_audio = BytesIO()
        sf.write(
            output_audio,
            denoised_audio,
            samplerate=self.config.SAMPLE_RATE,
            format=self.config.FORMAT
        )
        output_audio.seek(0)
        return output_audio

def predict_denoised_audio(
    audio_bytes: bytes, 
    model: TensorFlowModel
) -> DenoiseResult:
    """Main function to handle audio denoising process"""
    service = DenoiseService()
    return service.process_audio(audio_bytes, model)
