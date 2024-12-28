import torch
import librosa
import numpy as np
from core.settings import TTS_STOP_THRESHOLD
from text_to_speech.configs.audio_config import Text2SpeechAudioConfig
from core.utils.text2sequence.vn import VietnameseText2Sequence
from core.utils.text2sequence.en import EnglishText2Sequence
from speaker_verification.services.data_preprocess import preprocess_audio
from core.utils.processors.audio_processor import AudioPreprocessor


def get_encoded_speech(audio, speaker_verification_model):
    processed_audio, _, _ = preprocess_audio(audio)
    
    with torch.no_grad():
        encoded_speech = speaker_verification_model.model(processed_audio)
        
    return encoded_speech

def generate_magnitude(mag2mel_model, mel):
    mel = torch.tensor(np.array([mel]))
    mag_db = mag2mel_model(mel)
    
    return mag_db

def generate_speech(text_to_speech_model, text, audio, speaker_verification_model, mel2mag_nodel=None, languange="en"):
    t2seq = EnglishText2Sequence()
    
    if languange == "vi":
        t2seq = VietnameseText2Sequence()

    encoded_speech = get_encoded_speech(speaker_verification_model=speaker_verification_model, audio=audio)
    text_seq = torch.tensor(np.array([t2seq(text)]))
    
    _, mels, _ = text_to_speech_model.generate(text_seq, encoded_speech)
    mel = mels.detach().cpu().numpy()[0].T * 3
    
    while np.max(mel[:, -1]) < TTS_STOP_THRESHOLD:
        mel = mel[:, :-1]
    
    # mag_db = generate_magnitude(mel2mag_nodel, mel)
    processor = AudioPreprocessor(Text2SpeechAudioConfig)
    audio = processor.mel_to_audio(mel)
    # audio = processor.magnitude_db_to_audio(mag_db)
    
    return audio * 6
