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
from speaker_verification.models import LSTM_SPEAKER_ENCODER, TRANSFORMER_SPEAKER_ENCODER
from text_to_speech.services.synthesis import Synthesizer, _denormalize, hparams
from text_to_speech.models import EN_TACOTRON, MEL2MAG, VI_TACOTRON

en_synthsiser = Synthesizer(t2s_model=EN_TACOTRON.model, t2seq=EnglishText2Sequence())
vi_synthsiser = Synthesizer(t2s_model=VI_TACOTRON.model, t2seq=VietnameseText2Sequence())
    
def get_encoded_speech(audio, speaker_verification_model, model_type="lstm"):
    processed_audio, _, _ = preprocess_audio(audio, model_type=model_type)
    
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
    print(lang)
    
    if lang == "vi":
        
        encoded_speech = get_encoded_speech(speaker_verification_model=TRANSFORMER_SPEAKER_ENCODER, audio=audio, model_type="transformer")
        texts = text.split("\n")
        mels = vi_synthsiser.synthesize_spectrograms(texts, [encoded_speech.detach().numpy()[0]])
        mel = np.concatenate(mels, axis=1)
        inv_filter_bank_audio = vi_synthsiser.mel_to_audio_using_griffin_lim(mel)
        processor = AudioPreprocessor(Text2SpeechAudioConfig)
        mel = processor.audio_to_mel_db(inv_filter_bank_audio)
        pred_mag = MEL2MAG.model(torch.FloatTensor(np.array([mel.T], dtype=np.float64)))
        pred_audio = processor.magnitude_db_to_audio_using_griffin(pred_mag.detach().cpu().numpy()[0].T)
        np.save("./saved_mel.npy", mel)
        
        inv_filter_bank_audio_buffer = io.BytesIO()
        sf.write(inv_filter_bank_audio_buffer, inv_filter_bank_audio, samplerate=Text2SpeechAudioConfig.SAMPLE_RATE, format='WAV')
        inv_filter_bank_audio_buffer.seek(0)
        base64_inv_filter_bank_audio = base64.b64encode(inv_filter_bank_audio_buffer.read()).decode("utf-8")
        
        pred_audio_buffer = io.BytesIO()
        sf.write(pred_audio_buffer, pred_audio, samplerate=Text2SpeechAudioConfig.SAMPLE_RATE, format='WAV')
        pred_audio_buffer.seek(0)
        base64_pred_audio = base64.b64encode(pred_audio_buffer.read()).decode("utf-8")
        
        base64_mel_spec = base64.b64encode(gen_spec_buffer(mel, spec="Mel")).decode("utf-8")
        base64_inv_filter_bank_mag_spec = base64.b64encode(gen_spec_buffer(en_synthsiser.generate_magnitude_from_audio(inv_filter_bank_audio), spec="Mel")).decode("utf-8")
        base64_pred_mag_spec = base64.b64encode(gen_spec_buffer(pred_mag.detach().cpu().numpy()[0].T, spec="Mel")).decode("utf-8")
        lang = "vi"
    else:
        
        encoded_speech = get_encoded_speech(speaker_verification_model=LSTM_SPEAKER_ENCODER, audio=audio, model_type="lstm")
        texts = text.split("\n")
        mels = en_synthsiser.synthesize_spectrograms(texts, [encoded_speech.detach().numpy()[0]])
        mel = np.concatenate(mels, axis=1)
        inv_filter_bank_audio = en_synthsiser.mel_to_audio_using_griffin_lim(mel)
        processor = AudioPreprocessor(Text2SpeechAudioConfig)
        mel = processor.audio_to_mel_db(inv_filter_bank_audio)
        pred_mag = MEL2MAG.model(torch.FloatTensor(np.array([mel.T], dtype=np.float64)))
        pred_audio = processor.magnitude_db_to_audio_using_griffin(pred_mag.detach().cpu().numpy()[0].T)
        np.save("./saved_mel.npy", mel)
        
        inv_filter_bank_audio_buffer = io.BytesIO()
        sf.write(inv_filter_bank_audio_buffer, inv_filter_bank_audio, samplerate=Text2SpeechAudioConfig.SAMPLE_RATE, format='WAV')
        inv_filter_bank_audio_buffer.seek(0)
        base64_inv_filter_bank_audio = base64.b64encode(inv_filter_bank_audio_buffer.read()).decode("utf-8")
        
        pred_audio_buffer = io.BytesIO()
        sf.write(pred_audio_buffer, pred_audio, samplerate=Text2SpeechAudioConfig.SAMPLE_RATE, format='WAV')
        pred_audio_buffer.seek(0)
        base64_pred_audio = base64.b64encode(pred_audio_buffer.read()).decode("utf-8")
        
        base64_mel_spec = base64.b64encode(gen_spec_buffer(mel, spec="Mel")).decode("utf-8")
        base64_inv_filter_bank_mag_spec = base64.b64encode(gen_spec_buffer(en_synthsiser.generate_magnitude_from_audio(inv_filter_bank_audio), spec="Mel")).decode("utf-8")
        base64_pred_mag_spec = base64.b64encode(gen_spec_buffer(pred_mag.detach().cpu().numpy()[0].T, spec="Mel")).decode("utf-8")
        lang = "en"
        
    return {
        "base64_audio": base64_inv_filter_bank_audio,
        "base64_mel2mag_audio": base64_pred_audio,
        "base64_mel_spec": base64_mel_spec,
        "base64_mag_spec": base64_inv_filter_bank_mag_spec,
        "base64_mel2mag_mag_spec": base64_pred_mag_spec,
        "lang": lang
    }
