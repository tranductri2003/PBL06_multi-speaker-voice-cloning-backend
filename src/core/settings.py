import os
from dotenv import load_dotenv

load_dotenv()

MODEL_PATHS = {
    "LstmSpeakerEncoder": os.getenv("SPEAKER_LSTM_ENCODER_MODEL"),
    "TransformerSpeakerEncoder": os.getenv("SPEAKER_TRANSFORMER_ENCODER_MODEL"),
    "ModifiedUNet": os.getenv("VOICE_ENHANCEMENT_MODIFIED_UNET_MODEL"),
    "UNet": os.getenv("VOICE_ENHANCEMENT_UNET_MODEL"),
    "UNetPlusPlus": os.getenv("VOICE_ENHANCEMENT_UNET_PLUS_PLUS_MODEL"),
    "CNN50": os.getenv("VOICE_ENHANCEMENT_CNN50_MODEL"),
    "CNN100": os.getenv("VOICE_ENHANCEMENT_CNN100_MODEL"),
}

DEVICE = os.getenv("DEVICE", "cpu")
