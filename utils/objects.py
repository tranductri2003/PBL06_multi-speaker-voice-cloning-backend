"""
    A file define Utterance and Speaker models  
"""

from pathlib import Path
import logging
from io import BytesIO
import librosa
import numpy as np

from utils.audio_process import gen_melspectrogram

class Utterance(object):
    """
    A class to represent an audio utterance.

    This class handles loading audio files, processing them into raw samples or Mel spectrograms,
    and provides methods for augmenting and segmenting the audio.

    Attributes:
        id (str): Identifier for the utterance.
        raw_file (Path | BytesIO): Path or BytesIO object of the audio file.
        y (np.ndarray): Raw audio samples loaded from the audio file.
    """

    def __init__(self, id: str = None, raw_file: Path | BytesIO = None):
        """
        Initialize the Utterance with an identifier and audio file.

        Args:
            id (str, optional): Identifier for the utterance. Defaults to None.
            raw_file (Path | BytesIO, optional): Path or BytesIO object of the audio file. Defaults to None.
        """
        self.id = id
        self.raw_file = raw_file
        self.y = self.raw()

    def raw(self, sr=16000, augment=False):
        """
        Get the raw audio samples.

        Args:
            sr (int, optional): Sample rate for loading the audio. Defaults to 16000.
            augment (bool, optional): Whether to apply random amplitude augmentation. Defaults to False.

        Returns:
            np.ndarray: Normalized raw audio samples.

        Raises:
            Exception: If the audio file is empty.
        """
        y, sr = librosa.load(self.raw_file, sr=sr)
        if y.size == 0:
            raise ValueError('Empty audio')
        
        y = 0.95 * librosa.util.normalize(y)
        if augment:
            amplitude = np.random.uniform(low=0.3, high=1.0)
            y = y * amplitude
        return y

    def melspectrogram(self, sr=16000, n_fft=1024, hop_length=256, win_length=1024, n_mels=80):
        """
        Get the Mel spectrogram features.

        Args:
            sr (int, optional): Sample rate. Defaults to 16000.
            n_fft (int, optional): Number of FFT components. Defaults to 1024.
            hop_length (int, optional): Number of samples between frames. Defaults to 256.
            win_length (int, optional): Length of each windowed segment. Defaults to 1024.
            n_mels (int, optional): Number of Mel bands. Defaults to 80.

        Returns:
            np.ndarray: Mel spectrogram features.

        Raises:
            Exception: If loading the Mel spectrogram fails.
        """
        try:
            return gen_melspectrogram(self.y, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length, n_mels=n_mels)
        except Exception:
            logging.debug('Failed to load Mel spectrogram, raw file: %s', {self.raw_file})
            raise

    def random_raw_segment(self, seq_len):
        """
        Return a random audio segment of a specified length.

        Args:
            seq_len (int): Desired length of the audio segment.

        Returns:
            np.ndarray: A segment of raw audio samples.
        """
        res = self.y
        ylen = len(res)
        if ylen < seq_len:
            pad_left = (seq_len - ylen) // 2
            pad_right = seq_len - ylen - pad_left
            res = np.pad(res, ((pad_left, pad_right)), mode='reflect')
        elif ylen > seq_len:
            max_seq_start = ylen - seq_len
            seq_start = np.random.randint(0, max_seq_start)
            seq_end = seq_start + seq_len
            res = res[seq_start:seq_end]

        return res

    def random_mel_segment(self, seq_len):
        """
        Return a random Mel spectrogram segment of a specified length.

        Args:
            seq_len (int): Desired length of the Mel spectrogram segment.

        Returns:
            np.ndarray: A segment of Mel spectrogram features.
        """
        mel = self.melspectrogram()
        _, tempo_len = mel.shape
        if tempo_len < seq_len:
            pad_left = (seq_len - tempo_len) // 2
            pad_right = seq_len - tempo_len - pad_left
            mel = np.pad(mel, ((0, 0), (pad_left, pad_right)), mode='reflect')
        elif tempo_len > seq_len:
            max_seq_start = tempo_len - seq_len
            seq_start = np.random.randint(0, max_seq_start)
            seq_end = seq_start + seq_len
            mel = mel[:, seq_start:seq_end]
        return mel


class Speaker(object):
    """
    A class to represent a speaker and their associated utterances.

    This class manages the collection of utterances for a specific speaker, allowing
    for the addition of new utterances and retrieval of random utterances for processing.

    Attributes:
        id (str): Identifier for the speaker.
        utterances (list): List of `Utterance` objects associated with this speaker.
    """

    def __init__(self, id: str):
        """
        Initialize the Speaker with an identifier.

        Args:
            id (str): Identifier for the speaker.
        """
        self.id = id
        self.utterances = []

    def add_utterance(self, utterance: Utterance):
        """
        Add an utterance to this speaker.

        Args:
            utterance (Utterance): The `Utterance` object to be added to the speaker's collection.
        """
        self.utterances.append(utterance)

    def random_utterances(self, n):
        """
        Return n random utterances from this speaker.

        Args:
            n (int): The number of random utterances to retrieve.

        Returns:
            list: A list of `Utterance` objects selected randomly from the speaker's utterances.

        Raises:
            ValueError: If n is greater than the number of available utterances.
        """
        if n > len(self.utterances):
            raise ValueError("Requested number of utterances exceeds available utterances.")
        
        return [self.utterances[idx] for idx in np.random.randint(0, len(self.utterances), n)]
