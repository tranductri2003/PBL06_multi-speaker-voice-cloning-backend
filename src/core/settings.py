import os
from dotenv import load_dotenv

load_dotenv()

MODEL_PATHS = {
    "LstmSpeakerEncoder": os.getenv("SPEAKER_LSTM_ENCODER_MODEL"),
    "TransformerSpeakerEncoder": os.getenv("SPEAKER_TRANSFORMER_ENCODER_MODEL"),
}

DEVICE = os.getenv("DEVICE", "cpu")
