import torch
import librosa
import numpy as np
import io
from io import BytesIO
import base64
from core.settings import TTS_STOP_THRESHOLD
from text_to_speech.configs.audio_config import Text2SpeechAudioConfig
from core.utils.text2sequence.vn import VietnameseText2Sequence
from core.utils.text2sequence.en import EnglishText2Sequence
from speaker_verification.services.data_preprocess import preprocess_audio
from core.utils.processors.audio_processor import AudioPreprocessor
import matplotlib.pyplot as plt
import soundfile as sf
from core.settings import MODEL_PATHS
from speaker_verification.models import LSTM_SPEAKER_ENCODER
from text_to_speech.services.synthesis import Synthesizer
from text_to_speech.models import EN_TACOTRON, MEL2MAG
from voice_enhancement.services.visualization import SpectrogramVisualizer

en_synthsiser = Synthesizer(model=EN_TACOTRON.model)
    
def get_encoded_speech(audio, speaker_verification_model):
    processed_audio, _, _ = preprocess_audio(audio)
    
    with torch.no_grad():
        encoded_speech = speaker_verification_model.model(processed_audio)
        
    return encoded_speech

def gen_spec_buffer(data, spec="Magnitude"):
    buffer = io.BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(data, cmap='inferno', origin='lower')
    plt.xlabel('Time')
    plt.ylabel(f'{spec} Frequency')
    plt.title(f'{spec} Spectrogram')
    plt.colorbar(format='%2.0f')
    plt.savefig(buffer, format="png", dpi=100)
    buffer.seek(0)
    plt.close()
    return buffer.getvalue()

def generate_magnitude(mag2mel_model, mel):
    mel = torch.tensor(np.array([mel]))
    mag_db = mag2mel_model(mel)
    
    return mag_db

def generate_speech(text, audio, lang="en"):

    if lang == "en":
        
        encoded_speech = get_encoded_speech(speaker_verification_model=LSTM_SPEAKER_ENCODER, audio=audio)
        global en_synthsiser
        texts = text.split("\n")
        mels = en_synthsiser.synthesize_spectrograms(texts, [encoded_speech.detach().numpy()[0]])
        mel = np.concatenate(mels, axis=1)
        audio = en_synthsiser.mel_to_audio_using_griffin_lim(mel)
        # processor = AudioPreprocessor(Text2SpeechAudioConfig)
        # mel = processor.audio_to_mel_db(audio*500)
        # mags = MEL2MAG.model(torch.FloatTensor(np.array([mel.T])))
        # mag = mags.detach().numpy()[0]
        # audio = processor.magnitude_db_to_audio(mag.T)
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, audio, samplerate=Text2SpeechAudioConfig.SAMPLE_RATE, format='WAV')
        audio_buffer.seek(0)
        
        base64_audio = base64.b64encode(audio_buffer.read()).decode("utf-8")
        base64_mel_spec = base64.b64encode(gen_spec_buffer(mel)).decode("utf-8")
    return {
        "base64_audio": base64_audio,
        "base64_mel_spec": base64_mel_spec,
    }
