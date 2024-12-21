from text_to_speech.models.origin import TNTModel
from core.settings import MODEL_PATHS, DEVICE
from core.utils.model import PyTorchModel

# ORIGIN_TEXT_TO_SPEECH = DataParallel(TNTModel())
# checkpoint = torch.load(MODEL_PATHS["OriginTextToSpeech"], weights_only=False, map_location=DEVICE)
# ORIGIN_TEXT_TO_SPEECH.load_state_dict(checkpoint["model_state_dict"])

# ORIGIN_TEXT_TO_SPEECH = PyTorchModel(
#     model_class=TNTModel,
#     model_path=MODEL_PATHS["OriginTextToSpeech"],
#     device=DEVICE,
#     is_parallel=True
# )

ORIGIN_TEXT_TO_SPEECH = None

