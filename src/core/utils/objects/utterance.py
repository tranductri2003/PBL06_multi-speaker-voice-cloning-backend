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

    def compute_partial_slices(self, n_samples, partial_utterance_n_frames=80,
                           min_pad_coverage=0.75, overlap=0.5):
        samples_per_frame = int(self.processor.config.SAMPLE_RATE * self.processor.config.FRAME_SHIFT)
        n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
        frame_step = max(int(np.round(partial_utterance_n_frames * (1 - overlap))), 1)
    
        wav_slices, mel_slices = [], []
        for i in range(0, n_frames, frame_step):
            mel_start = i
            mel_end = mel_start + partial_utterance_n_frames
    
            if mel_end > n_frames:  # Adjust for slices that exceed available frames
                mel_start = max(0, n_frames - partial_utterance_n_frames)
                mel_end = n_frames
    
            if mel_end - mel_start == partial_utterance_n_frames:  # Validate slice size
                mel_slices.append(slice(mel_start, mel_end))
                wav_slices.append(slice(mel_start * samples_per_frame, mel_end * samples_per_frame))
    
        return wav_slices, mel_slices

    

    def split_mel_into_frames(self, partial_utterance_n_frames=160, overlap=0.5):
        mel_spectrogram = self.mel_in_db()
        n_samples = len(self.audio)
        _, mel_slices = self.compute_partial_slices(n_samples, 
                                                    partial_utterance_n_frames=partial_utterance_n_frames, 
                                                    overlap=overlap)
        mel_frames = []
        for s in mel_slices:
            mel_frame = mel_spectrogram[:, s]
            if mel_frame.shape[1] == partial_utterance_n_frames:
                mel_frames.append(mel_frame)
            elif mel_frame.shape[1] > 0.9 * partial_utterance_n_frames:
                padding = partial_utterance_n_frames - mel_frame.shape[1]
                pad_left = padding // 2
                pad_right = padding - pad_left
                mel_frame = np.pad(mel_frame, ((0, 0), (pad_left, pad_right)), mode="constant")
                mel_frames.append(mel_frame)
        return mel_frames