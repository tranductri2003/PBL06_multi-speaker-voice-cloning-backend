from io import BytesIO
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
from typing import Tuple
from ..configs.voice_enhancement_config import VoiceEnhancementConfig

class SpectrogramVisualizer:
    def __init__(self, config: VoiceEnhancementConfig = VoiceEnhancementConfig()):
        self.config = config

    def save_spectrogram(self, data: np.ndarray, title: str) -> bytes:
        """Generate and save spectrogram visualization"""
        buffer = BytesIO()
        plt.figure(figsize=self.config.SPEC_FIGSIZE)
        librosa.display.specshow(
            data, 
            sr=self.config.SAMPLE_RATE, 
            hop_length=self.config.HOP_LENGTH_FFT, 
            x_axis='time', 
            y_axis='linear'
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(buffer, format="png", dpi=self.config.DPI)
        buffer.seek(0)
        plt.close()
        return buffer.getvalue()

    def save_waveplot(self, audio_data: np.ndarray, title: str) -> bytes:
        """Generate and save waveform visualization"""
        buffer = BytesIO()
        plt.figure(figsize=self.config.WAVE_FIGSIZE)
        librosa.display.waveshow(
            audio_data, 
            sr=self.config.SAMPLE_RATE, 
            x_axis='time', 
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(buffer, format="png", dpi=self.config.DPI)
        buffer.seek(0)
        plt.close()
        return buffer.getvalue()

    def save_phase_plot(self, stft_phase: np.ndarray, title: str) -> bytes:
        """Generate and save phase visualization"""
        buffer = BytesIO()
        plt.figure(figsize=self.config.SPEC_FIGSIZE)
        librosa.display.specshow(
            np.angle(stft_phase),
            sr=self.config.SAMPLE_RATE,
            hop_length=self.config.HOP_LENGTH_FFT,
            x_axis='time',
            y_axis='linear'
        )
        plt.colorbar()
        plt.title(title)
        plt.tight_layout()
        plt.savefig(buffer, format="png", dpi=self.config.DPI)
        buffer.seek(0)
        plt.close()
        return buffer.getvalue()

    def generate_visualizations(
        self,
        m_amp_db: np.ndarray,
        inv_sca_X_pred: np.ndarray,
        X_denoise: np.ndarray,
        original_audio: np.ndarray,
        noised_audio: np.ndarray,
        denoised_audio: np.ndarray,
    ) -> Tuple[bytes, bytes, bytes, bytes, bytes, bytes]:
        """Generate all visualizations for the denoising process"""
        return (
            # Spectrograms
            self.save_spectrogram(
                m_amp_db,
                "Original Spectrogram (Voice + Noise)"
            ),
            self.save_spectrogram(
                inv_sca_X_pred,
                "Predicted Noise Spectrogram"
            ),
            self.save_spectrogram(
                X_denoise,
                "Denoised Voice Spectrogram"
            ),
            # Waveforms
            self.save_waveplot(
                original_audio,
                "Original Waveform (Voice + Noise)"
            ),
            self.save_waveplot(
                noised_audio,
                "Predicted Noise Waveform"
            ),
            self.save_waveplot(
                denoised_audio,
                "Denoised Voice Waveform"
            )
        )

    def plot_comparison(
        self,
        voice_noise: np.ndarray,
        predicted_noise: np.ndarray,
        denoised_voice: np.ndarray,
        plot_type: str = "spectrogram"
    ) -> bytes:
        """Generate comparison plot of original, noise and denoised audio"""
        buffer = BytesIO()
        plt.figure(figsize=(8, 12))
        
        if plot_type == "spectrogram":
            plot_func = self.save_spectrogram
            titles = ["Voice + Noise", "Predicted Noise", "Denoised Voice"]
        elif plot_type == "phase":
            plot_func = self.save_phase_plot
            titles = ["Phase (Voice + Noise)", "Phase (Predicted Noise)", "Phase (Denoised Voice)"]
        else:  # waveform
            plot_func = self.save_waveplot
            titles = ["Waveform (Voice + Noise)", "Waveform (Predicted Noise)", "Waveform (Denoised Voice)"]

        # Plot three subplots
        for i, (data, title) in enumerate(zip([voice_noise, predicted_noise, denoised_voice], titles), 1):
            plt.subplot(3, 1, i)
            plot_func(data, title)
            
        plt.tight_layout()
        plt.savefig(buffer, format="png", dpi=self.config.DPI)
        buffer.seek(0)
        plt.close()
        return buffer.getvalue()
    