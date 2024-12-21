import librosa
import numpy as np
from core.utils.configs.audio import AudioConfig


class AudioPreprocessor:
    def __init__(self, config: AudioConfig):
        self.config = config

    def normalize(self, spectrogram_in_db):
        normalized_spectrogram_in_db = (
            spectrogram_in_db - self.config.REF_LEVEL_DB - self.config.MIN_LEVEL_DB
        ) / -self.config.MIN_LEVEL_DB

        return np.clip(normalized_spectrogram_in_db, self.config.ZERO_THRESHOLD, 1)

    def magnitude_to_mel(self, magnitude):
        return librosa.feature.melspectrogram(
            S=magnitude,
            sr=self.config.SAMPLE_RATE,
            n_fft=self.config.N_FFT,
            n_mels=self.config.N_MELS,
            fmin=self.config.FMIN,
            fmax=self.config.FMAX,
        )

    def amp_to_db(self, mel_spectrogram):
        return 20.0 * np.log10(
            np.maximum(self.config.ZERO_THRESHOLD, mel_spectrogram)
        )

    def audio_to_stft(self, audio):
        return librosa.stft(
            y=audio,
            n_fft=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH,
            win_length=self.config.WIN_LENGTH,
        )

    def apply_pre_emphasis(self, y):
        return np.append(y[0], y[1:] - self.config.PRE_EMPHASIS * y[:-1])

    def stft_to_magnitude(self, linear):
        return np.abs(linear)

    def audio_to_mel_db(self, audio):
        stft = self.audio_to_stft(audio)
        magnitude = self.stft_to_magnitude(stft)
        mel = self.magnitude_to_mel(magnitude)
        mel = self.amp_to_db(mel)
        return self.normalize(mel)
    
    def audio_to_magnitude_db(self, audio):
        stft = self.audio_to_stft(audio)
        magnitude_in_amp =  self.stft_to_magnitude(stft)
        magnitude_in_db = self.amp_to_db(magnitude_in_amp)
        return self.normalize(magnitude_in_db)

    def magnitude_to_stft(self, magnitude_in_amp):
        # Use a random phase
        random_phase = np.random.uniform(-np.pi, np.pi, size=magnitude_in_amp.shape)
        return magnitude_in_amp * np.exp(1j * random_phase)

    def denormalize(self, spectrogram_in_db):
        denormalized_spectrogram_in_db = (
            spectrogram_in_db * -self.config.MIN_LEVEL_DB
            + self.config.REF_LEVEL_DB + self.config.MIN_LEVEL_DB
        )
        return denormalized_spectrogram_in_db
    
    def db_to_amp(self, magnitude_in_db):
        return 10 ** (magnitude_in_db / 20)
    
    def magnitude_db_to_audio(self, magnitude_in_db):
        magnitude_in_db = self.denormalize(magnitude_in_db)
        magnitude_in_amp = self.db_to_amp(magnitude_in_db)
        stft = self.magnitude_to_stft(magnitude_in_amp)
        audio = self.stft_to_audio(stft)
        return audio
    
    def stft_to_audio(self, stft):
        return librosa.istft(
            stft,
            hop_length=self.config.HOP_LENGTH,
            win_length=self.config.WIN_LENGTH,
        )

    def mel_to_audio(self, mel):
        stft = librosa.feature.inverse.mel_to_stft(mel, sr=16000)
        return self.stft_to_audio(stft)

    def griffin(self, stft):
        return  librosa.griffinlim(stft)

    def magnitude_db_to_audio_using_griffin(self, magnitude_in_db):
        magnitude_in_db = self.denormalize(magnitude_in_db)
        magnitude_in_amp = self.db_to_amp(magnitude_in_db)
        stft = self.magnitude_to_stft(magnitude_in_amp)
        audio = self.griffin(stft)
        return audio
    