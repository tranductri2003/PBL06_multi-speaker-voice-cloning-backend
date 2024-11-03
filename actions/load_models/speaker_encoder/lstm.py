import torch
from utils.models.speaker_encoder import SpeakerLstmEncoder


def load_speaker_lstm_encoder(model_settings):
    model = SpeakerLstmEncoder(
        device=model_settings.DEVICE, loss_device=model_settings.DEVICE
    )
    ckpt = torch.load(model_settings.MODEL_PATH, weights_only=False)

    if ckpt:
        model.load_state_dict(ckpt["model_state_dict"])

    model.eval()
    return model
