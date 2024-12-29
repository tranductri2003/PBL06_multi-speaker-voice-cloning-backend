from text_to_speech.models.tacotron import Tacotron
from text_to_speech.models.mel2mag import ModelPostNet
from core.settings import MODEL_PATHS, DEVICE, EN_TACOTRON_PARAMS, VI_TACOTRON_PARAMS
from core.utils.model import PyTorchModel

EN_TACOTRON = PyTorchModel(
    model_class=Tacotron,
    model_path=MODEL_PATHS["EN_TACOTRON"],
    device=DEVICE,
    is_parallel=True,
    model_params=EN_TACOTRON_PARAMS,
    model_state="model_state"
)

VI_TACOTRON = PyTorchModel(
    model_class=Tacotron,
    model_path=MODEL_PATHS["VI_TACOTRON"],
    device=DEVICE,
    is_parallel=True,
    model_params=VI_TACOTRON_PARAMS,
    model_state="model_state_dict"
)

MEL2MAG = PyTorchModel(
    model_class=ModelPostNet,
    model_path=MODEL_PATHS["Mel2Mag"],
    device=DEVICE,
    is_parallel=False,
    model_state="model"
)