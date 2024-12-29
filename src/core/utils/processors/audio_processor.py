import librosa
import numpy as np


class AudioPreprocessor:
    def __init__(self, config):
        self.config = config

    def normalize(self, spectrogram_in_db):
        '''Normalize spectrogram in decibel values between 0 and 1.'''
        normalized_spectrogram_in_db = (
            spectrogram_in_db - self.config.REF_LEVEL_DB - self.config.MIN_LEVEL_DB
        ) / -self.config.MIN_LEVEL_DB

        return np.clip(normalized_spectrogram_in_db, self.config.ZERO_THRESHOLD, 1)

    def magnitude_to_mel(self, magnitude):
        '''Convert a magnitude spectrogram to a mel spectrogram.'''
        return librosa.feature.melspectrogram(
            S=magnitude,
            sr=self.config.SAMPLE_RATE,
            n_fft=self.config.N_FFT,
            n_mels=self.config.N_MELS,
            fmin=self.config.FMIN,
            fmax=self.config.FMAX,
        )

    def amp_to_db(self, mel_spectrogram):
        '''Convert amplitude spectrogram to decibel scale.'''
        return 20.0 * np.log10(
            np.maximum(self.config.ZERO_THRESHOLD, mel_spectrogram)
        )

    def audio_to_stft(self, audio):
        '''Generate Short-Time Fourier Transform (STFT) from the audio time series.'''
        return librosa.stft(
            y=audio,
            n_fft=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH,
            win_length=self.config.WIN_LENGTH,
        )

    def apply_pre_emphasis(self, y):
        '''Apply a pre-emphasis filter to the audio signal.'''
        return np.append(y[0], y[1:] - self.config.PRE_EMPHASIS * y[:-1])

    def stft_to_magnitude(self, linear):
        '''Compute the magnitude spectrogram from STFT.'''
        return np.abs(linear)

    def audio_to_mel_db(self, audio):
        '''Convert a given linear spectrogram to a log mel spectrogram (mel spectrogram in db) and return it.'''
        stft = self.audio_to_stft(audio)
        magnitude = self.stft_to_magnitude(stft)
        mel = self.magnitude_to_mel(magnitude)
        mel = self.amp_to_db(mel)
        return self.normalize(mel)
    
    def audio_to_magnitude_db(self, audio):
        '''Convert a given linear spectrogram to a magnitude spectrogram.'''
        stft = self.audio_to_stft(audio)
        magnitude_in_amp =  self.stft_to_magnitude(stft)
        magnitude_in_db = self.amp_to_db(magnitude_in_amp)
        return self.normalize(magnitude_in_db)

    def magnitude_to_stft(self, magnitude_in_amp):
        '''Compute the STFT from the magnitude spectrogram.'''
        # Use a random phase
        random_phase = np.random.uniform(-np.pi, np.pi, size=magnitude_in_amp.shape)
        return magnitude_in_amp * np.exp(1j * random_phase)

    def denormalize(self, spectrogram_in_db):
        '''Denormalize spectrogram in decibel values.'''
        denormalized_spectrogram_in_db = (
            spectrogram_in_db * -self.config.MIN_LEVEL_DB
            + self.config.REF_LEVEL_DB + self.config.MIN_LEVEL_DB
        )
        return denormalized_spectrogram_in_db
    
    def db_to_amp(self, magnitude_in_db):
        '''Convert decibel scale to amplitude.'''
        return 10 ** (magnitude_in_db / 20)
    
    def magnitude_db_to_audio(self, magnitude_in_db):
        '''Convert a given magnitude spectrogram in dB to a linear audio signal.'''
        magnitude_in_db = self.denormalize(magnitude_in_db)
        magnitude_in_amp = self.db_to_amp(magnitude_in_db)
        stft = self.magnitude_to_stft(magnitude_in_amp)
        audio = self.stft_to_audio(stft)
        return audio
    
    def stft_to_audio(self, stft):
        '''Generate audio time series from the STFT.'''
        return librosa.istft(
            stft,
            hop_length=self.config.HOP_LENGTH,
            win_length=self.config.WIN_LENGTH,
        )

    def mel_to_audio(self, mel):
        '''Generate audio time series from the STFT.'''
        stft = librosa.feature.inverse.mel_to_stft(mel, sr=16000)
        return self.stft_to_audio(stft)

    def griffin(self, stft):
        return  librosa.griffinlim(stft)

    def magnitude_db_to_audio_using_griffin(self, magnitude_in_db):
        '''Convert a given magnitude spectrogram in dB to a linear audio signal.'''
        magnitude_in_db = self.denormalize(magnitude_in_db)
        magnitude_in_amp = self.db_to_amp(magnitude_in_db)
        stft = self.magnitude_to_stft(magnitude_in_amp)
        audio = self.griffin(stft)
        return audio
    