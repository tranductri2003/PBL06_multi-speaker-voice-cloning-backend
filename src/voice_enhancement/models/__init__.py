from core.utils.model import TensorFlowModel
from core.settings import MODEL_PATHS

# Initialize models
MODIFIED_UNET = TensorFlowModel(
    model_path=MODEL_PATHS["ModifiedUNet"]
)

UNET = TensorFlowModel(
    model_path=MODEL_PATHS["UNet"]
)

UNET_PLUS_PLUS = TensorFlowModel(
    model_path=MODEL_PATHS["UNetPlusPlus"]
)

CNN50 = TensorFlowModel(
    model_path=MODEL_PATHS["CNN50"]
)

CNN100 = TensorFlowModel(
    model_path=MODEL_PATHS["CNN100"]
)

# Model mapping
MODEL_MAPPING = {
    "modified_unet": MODIFIED_UNET,
    "unet": UNET,
    "unet_plus_plus": UNET_PLUS_PLUS,
    "cnn50": CNN50,
    "cnn100": CNN100
}

# Import manager after constants are defined
from .manager import VoiceEnhancementModelManager

__all__ = [
    'MODIFIED_UNET',
    'UNET',
    'UNET_PLUS_PLUS',
    'CNN50',
    'CNN100',
    'MODEL_MAPPING',
    'VoiceEnhancementModelManager'
] 