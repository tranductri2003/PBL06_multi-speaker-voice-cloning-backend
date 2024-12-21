from text_to_speech.models.origin import TNTModel
from text_to_speech.models.mel2mag import ModelPostNet
from core.settings import MODEL_PATHS, DEVICE
from core.utils.model import PyTorchModel

ORIGIN_TEXT_TO_SPEECH = PyTorchModel(
    model_class=TNTModel,
    model_path=MODEL_PATHS["OriginTextToSpeech"],
    device=DEVICE,
    is_parallel=True
)

MEL2MAG = PyTorchModel(
    model_class=ModelPostNet,
    model_path=MODEL_PATHS["Mel2Mag"],
    device=DEVICE,
    is_parallel=False,
    model_state="model"
)