from pathlib import Path
import logging
from io import BytesIO
import librosa
import numpy as np
from core.utils.processors.audio_processor import AudioPreprocessor


class Utterance(object):

    def __init__(
        self,
        _id: str = None,
        raw_file: Path | BytesIO = None,
        processor=None,
    ):
        self._id = _id
        self.raw_file = raw_file
        self.processor = processor if processor is not None else AudioPreprocessor()
        self.audio = self.raw()

    def raw(self):
        if isinstance(self.raw_file, Path) and self.raw_file.endswith(".npy"):
            return np.load(self.raw_file)

        audio, _ = librosa.load(self.raw_file, sr=self.processor.config.SAMPLE_RATE)

        if audio.size == 0:
            raise ValueError("Empty audio")

        audio = self.processor.config.SCALING_FACTOR * librosa.util.normalize(audio)
        return audio

    def mel_in_db(self):
        try:
            return self.processor.audio_to_mel_db(self.audio)
        except Exception:

            logging.debug(
                "Failed to load Mel spectrogram, raw file: %s", {self.raw_file}
            )
            raise

    def random_mel_in_db(self, num_frames=128):
        random_mel = self.mel_in_db()
        _, tempo_len = random_mel.shape
        if tempo_len < num_frames:
            pad_left = (num_frames - tempo_len) // 2
            pad_right = num_frames - tempo_len - pad_left
            random_mel = np.pad(
                random_mel, ((0, 0), (pad_left, pad_right)), mode="reflect"
            )
        elif tempo_len > num_frames:
            max_seq_start = tempo_len - num_frames
            seq_start = np.random.randint(0, max_seq_start)
            seq_end = seq_start + num_frames
            random_mel = random_mel[:, seq_start:seq_end]
        return random_mel

    def magtitude(self):
        return self.processor.audio_to_magnitude_db(self.audio)
