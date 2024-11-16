import os
from dotenv import load_dotenv
import numpy as np
import soundfile as sf
import librosa
import tensorflow as tf
from .process_func import inv_scaled_ou, matrix_spectrogram_to_numpy_audio
from .process_audio import preprocess_audio_for_prediction
from io import BytesIO
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

def save_spectrogram_image(data, sample_rate, hop_length, title):
    buffer = BytesIO()
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(data, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plt.close()
    return buffer.getvalue()

def save_waveplot_image(audio_data, sample_rate, title="Waveplot of Audio"):
    buffer = BytesIO()
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(audio_data, sr=sample_rate, x_axis='time', color='b')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plt.close()
    return buffer.getvalue()


def predict_denoised_audio(
    audio_bytes,
    model_name: str,
    frame_length: int = 8064,
    hop_length_fft: int = 63,
    sample_rate: int = 8000,
):
    model_path = os.getenv("MODEL_PATH")
    # Load the model
    loaded_model = tf.keras.models.load_model(model_path + model_name + ".keras")
    
    raw_audio, _ = librosa.load(BytesIO(audio_bytes), sr=sample_rate)
    
    X_in, m_amp_db_audio, m_pha_audio = preprocess_audio_for_prediction(raw_audio)
    
    # Make predictions
    X_pred = loaded_model.predict(X_in)

    # Rescale back to original
    inv_sca_X_pred = inv_scaled_ou(X_pred)

    # Denoising
    X_denoise = m_amp_db_audio - inv_sca_X_pred[:, :, :, 0]

    # Reconstruct audio
    audio_denoise_recons = matrix_spectrogram_to_numpy_audio(X_denoise, m_pha_audio, frame_length, hop_length_fft)
    audio_denoise_recons *= 6.0

    denoised_audio = np.concatenate(audio_denoise_recons, axis=0)

    # Prepare audio
    output_audio = BytesIO()
    sf.write(output_audio, denoised_audio, samplerate=8000, format="WAV")
    output_audio.seek(0)

    # Prepare spectrogram images
    spectrogram_m_amp_db = save_spectrogram_image(
        m_amp_db_audio[0], sample_rate, hop_length_fft, "Original Spectrogram (m_amp_db_audio)"
    )
    spectrogram_inv_sca_X_pred = save_spectrogram_image(
        inv_sca_X_pred[0, :, :, 0], sample_rate, hop_length_fft, "Predicted Spectrogram (inv_sca_X_pred)"
    )
    spectrogram_X_denoise = save_spectrogram_image(
        X_denoise[0], sample_rate, hop_length_fft, "Denoised Spectrogram (X_denoise)"
    )

    wave_m_amp_db = save_waveplot_image(
        m_amp_db_audio[0], sample_rate,"Original Spectrogram (m_amp_db_audio)"
    )
    wave_inv_sca_X_pred = save_waveplot_image(
        inv_sca_X_pred[0, :, :, 0], sample_rate,"Predicted Spectrogram (inv_sca_X_pred)"
    )
    wave_X_denoise = save_waveplot_image(
        X_denoise[0], sample_rate,"Denoised Spectrogram (X_denoise)"
    )

    return output_audio, spectrogram_m_amp_db, spectrogram_inv_sca_X_pred, spectrogram_X_denoise, wave_m_amp_db, wave_inv_sca_X_pred, wave_X_denoise