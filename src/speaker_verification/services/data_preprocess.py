from io import BytesIO
from pathlib import Path
import torch
import numpy as np

from speaker_verification.configs.audio_config import SpeakerEncoderAudioConfig
from core.utils.objects.utterance import Utterance
from core.utils.processors.audio_processor import AudioPreprocessor


def preprocess_audio(file: BytesIO | str | Path) -> torch.Tensor:
    uttn = Utterance(
        raw_file=file, processor=AudioPreprocessor(config=SpeakerEncoderAudioConfig)
    )

    return torch.tensor(
        np.array(
            [uttn.random_mel_in_db(num_frames=SpeakerEncoderAudioConfig.NUM_FRAMES)]
        )
    ).transpose(1, 2)
