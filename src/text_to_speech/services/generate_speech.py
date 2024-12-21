import torch
import librosa
import  copy
import numpy as np
from scipy import signal
from text_to_speech.configs.audio_config import Text2SpeechAudioConfig
from core.utils.tokenizer.text_to_sequence import WordByPhonemesEmbedding
from speaker_verification.services.data_preprocess import preprocess_audio
def get_encoded_speech(audio, speaker_verification_model):
    processed_audio, _, _ = preprocess_audio(audio)
    with torch.no_grad():
        encoded_speech = speaker_verification_model.model(processed_audio)
    return encoded_speech

def generate_speech(text_to_speech_model, text, audio, speaker_verification_model):
    t2seq = WordByPhonemesEmbedding()
    text_sequence = torch.FloatTensor(np.array([t2seq(text)]))

    text_length = len(text_sequence) 
    encoded_speech = get_encoded_speech(speaker_verification_model=speaker_verification_model, audio=audio)
    text_to_speech_model.eval()
    t2seq = WordByPhonemesEmbedding()

    text_sequence = torch.FloatTensor(np.array([t2seq(text)]))
    mel_input = torch.FloatTensor(np.array([np.zeros([1, Text2SpeechAudioConfig.N_MELS], np.float32)]))
                                      
    pos_text = torch.LongTensor(np.array([np.arange(1, text_length + 1)]))

    with torch.no_grad():
        for i in range(1000):
            pos_audio = torch.arange(1, mel_input.size(1) + 1).unsqueeze(0)
            mel_pred, postnet_ouput = text_to_speech_model(text_sequence, pos_text, mel_input, pos_audio, encoded_speech)
            mel_input = torch.concat([torch.FloatTensor(np.array([np.zeros([1, Text2SpeechAudioConfig.N_MELS], np.float32)])), mel_pred], dim=1)
    
    mel = postnet_ouput.detach().cpu().numpy()[0].T
    amplitude_mel = librosa.db_to_amplitude(mel)

    # Convert mel to linear spectrogram
    linear_spectrogram = librosa.feature.inverse.mel_to_stft(
        amplitude_mel,
        sr=Text2SpeechAudioConfig.SAMPLE_RATE,
        n_fft=Text2SpeechAudioConfig.N_FFT
    )
    print(f"Linear spectrogram shape: {linear_spectrogram.shape}")

    # Reconstruct audio using Griffin-Lim
    audio = librosa.griffinlim(
        linear_spectrogram,
        hop_length=Text2SpeechAudioConfig.HOP_LENGTH,
        win_length=Text2SpeechAudioConfig.WIN_LENGTH,
        n_iter=32
    )
    return audio * 6

def spectrogram_to_wav(mag):
    mag = mag.T

    mag = (np.clip(mag, 0, 1) * Text2SpeechAudioConfig.MAX_DB) - Text2SpeechAudioConfig.MAX_DB + Text2SpeechAudioConfig.REF_LEVEL_DB

    mag = np.power(10.0, mag * 0.05)

    wav = griffin_lim(mag**Text2SpeechAudioConfig.POWER)

    wav = signal.lfilter([1], [1, -Text2SpeechAudioConfig.PRE_EMPHASIS], wav)

    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)

def griffin_lim(spectrogram):
    X_best = copy.deepcopy(spectrogram)
    for i in range(Text2SpeechAudioConfig.N_ITER):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, Text2SpeechAudioConfig.N_FFT, Text2SpeechAudioConfig.HOP_LENGTH, win_length=Text2SpeechAudioConfig.WIN_LENGTH)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)

    return y
def invert_spectrogram(spectrogram):
    return librosa.istft(spectrogram, Text2SpeechAudioConfig.HOP_LENGTH, win_length=Text2SpeechAudioConfig.WIN_LENGTH, window="hann")