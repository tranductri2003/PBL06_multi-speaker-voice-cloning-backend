import os
from dotenv import load_dotenv
from actions.load_models import (
    load_speaker_transformer_encoder,
    load_speaker_lstm_encoder,
)
from utils.configs import SpeakerEncoderAudioConfig
from utils.processors import AudioPreprocessor

load_dotenv()


class SpeakerEncoderAudioSettings:

    AUDIO_CONFIG = SpeakerEncoderAudioConfig
    AUDIO_PROCESSOR = AudioPreprocessor(AUDIO_CONFIG)
    NUM_ITERATIONS = 33


class SpeakerTransformerEncoderModelSettings:
    DEVICE = os.getenv("DEVICE", "cpu")
    MODEL_PATH = os.getenv("SPEAKER_TRANSFORMER_ENCODER_MODEL", None)


SPEAKER_TRANSFORMER_ENCODER = load_speaker_transformer_encoder(
    SpeakerTransformerEncoderModelSettings
)

class SpeakerLstmEncoderModelSettings:
    DEVICE = os.getenv("DEVICE", "cpu")
    MODEL_PATH = os.getenv("SPEAKER_LSTM_ENCODER_MODEL", None)


SPEAKER_LSTM_ENCODER = load_speaker_lstm_encoder(
    SpeakerLstmEncoderModelSettings
)

TEST_FOLDER = os.getenv("TEST_FOLDER", None)

