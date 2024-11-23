from speaker_verification.models.lstm import LstmSpeakerEncoder
from speaker_verification.models.transformer import TransformerSpeakerEncoder

from core.utils.model import PyTorchModel
from core.settings import MODEL_PATHS, DEVICE

LSTM_SPEAKER_ENCODER = PyTorchModel(
    model_class=LstmSpeakerEncoder,
    model_path=MODEL_PATHS["LstmSpeakerEncoder"],
    device=DEVICE,
)
TRANSFORMER_SPEAKER_ENCODER = PyTorchModel(
    model_class=TransformerSpeakerEncoder,
    model_path=MODEL_PATHS["TransformerSpeakerEncoder"],
    device=DEVICE,
)
