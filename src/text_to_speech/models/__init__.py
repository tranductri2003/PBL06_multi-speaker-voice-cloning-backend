from text_to_speech.models.origin import TNTModel
from core.utils.model import PytorchModel
from core.settings import MODEL_PATHS, DEVICE
import os
import torch
from torch.nn import DataParallel

ORIGIN_TEXT_TO_SPEECH = DataParallel(TNTModel())
checkpoint = torch.load(MODEL_PATHS["OriginTextToSpeech"], weights_only=False, map_location=DEVICE)
ORIGIN_TEXT_TO_SPEECH.load_state_dict(checkpoint["model_state_dict"])

# ORIGIN_TEXT_TO_SPEECH = PytorchModel(
#     model_class=TNTModel,
#     model_path=MODEL_PATHS["OriginTextToSpeech"],
#     device=DEVICE,
# )


