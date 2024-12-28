import torch
import librosa
import numpy as np
from core.settings import TTS_STOP_THRESHOLD
from text_to_speech.configs.audio_config import Text2SpeechAudioConfig
from core.utils.text2sequence.vn import VietnameseText2Sequence
from core.utils.text2sequence.en import EnglishText2Sequence
from speaker_verification.services.data_preprocess import preprocess_audio
from core.utils.processors.audio_processor import AudioPreprocessor
import matplotlib.pyplot as plt
import soundfile as sf
from core.settings import MODEL_PATHS
from text_to_speech.services.synthesis import Synthesizer
from text_to_speech.models import EN_TACOTRON

en_synthsiser = Synthesizer(model=EN_TACOTRON.model)
    
def get_encoded_speech(audio, speaker_verification_model):
    processed_audio, _, _ = preprocess_audio(audio)
    
    with torch.no_grad():
        encoded_speech = speaker_verification_model.model(processed_audio)
        
    return encoded_speech

def generate_magnitude(mag2mel_model, mel):
    mel = torch.tensor(np.array([mel]))
    mag_db = mag2mel_model(mel)
    
    return mag_db

def generate_speech(text, audio, speaker_verification_model, mel2mag_nodel=None, languange="en"):

    if languange == "en":
        
        encoded_speech = get_encoded_speech(speaker_verification_model=speaker_verification_model, audio=audio)
        global en_synthsiser
        texts = text.split("\n")
        mels = en_synthsiser.synthesize_spectrograms(texts, [encoded_speech.detach().numpy()[0]])
        mel = np.concatenate(mels, axis=1)
        audio = en_synthsiser.griffin_lim(mel)
        sf.write("./generated_audio.wav", audio*6, 16000)

    return audio * 6
