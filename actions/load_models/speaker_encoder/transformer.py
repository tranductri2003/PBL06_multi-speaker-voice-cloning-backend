import torch
from utils.models.speaker_encoder import SpeakerTransformerEncoder


def load_speaker_transformer_encoder(model_settings):
    model = SpeakerTransformerEncoder(
        device=model_settings.DEVICE, loss_device=model_settings.DEVICE
    )
    ckpt = torch.load(model_settings.MODEL_PATH, weights_only=False, map_location=torch.device(model_settings.DEVICE))

    if ckpt:
        model.load_state_dict(ckpt["model_state_dict"])

    model.eval()
    return model
