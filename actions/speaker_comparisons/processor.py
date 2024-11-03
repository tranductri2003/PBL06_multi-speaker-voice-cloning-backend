from io import BytesIO
from pathlib import Path
import torch

import numpy as np
from utils.objects import Utterance
from settings import SpeakerEncoderAudioSettings


def preprocess_audio(file: BytesIO | str | Path) -> torch.Tensor:
    uttn = Utterance(
        raw_file=file, processor=SpeakerEncoderAudioSettings.AUDIO_PROCESSOR
    )

    return torch.tensor(
        np.array(
            [
                uttn.random_mel_in_db(
                    num_frames=SpeakerEncoderAudioSettings.AUDIO_CONFIG.NUM_FRAMES
                )
                for _ in range(SpeakerEncoderAudioSettings.NUM_ITERATIONS)
            ]
        )
    ).transpose(1, 2)
