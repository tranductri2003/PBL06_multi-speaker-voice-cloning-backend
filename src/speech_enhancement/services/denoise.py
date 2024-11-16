import numpy as np
import soundfile as sf
import tensorflow as tf
from .process_func import inv_scaled_ou, matrix_spectrogram_to_numpy_audio
from io import BytesIO

def predict_denoised_audio(
    model_path: str,
    X_in: np.ndarray,
    m_amp_db_audio: np.ndarray,
    m_pha_audio: np.ndarray,
    frame_length: int,
    hop_length_fft: int,
    sample_rate: int = 8000
) -> bytes:

    # Load model
    loaded_model = tf.keras.models.load_model(model_path)

    # Make predictions
    X_pred = loaded_model.predict(X_in)

    # Rescale back to original
    inv_sca_X_pred = inv_scaled_ou(X_pred)

    # Denoising
    X_denoise = m_amp_db_audio - inv_sca_X_pred[:, :, :, 0]

    # Reconstruct audio
    audio_denoise_recons = matrix_spectrogram_to_numpy_audio(X_denoise, m_pha_audio, frame_length, hop_length_fft)
    audio_denoise_recons *= 6.0

    # Save denoised audio to bytes
    audio_bytes = sf.write(BytesIO(), audio_denoise_recons.flatten(), sample_rate, format='WAV')
    print(f"Amplitude after denoising: {np.max(X_denoise)}")

    return audio_bytes
