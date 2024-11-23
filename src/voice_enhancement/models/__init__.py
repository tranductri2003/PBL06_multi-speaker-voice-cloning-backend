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

# Model mapping
MODEL_MAPPING = {
    "modified_unet": MODIFIED_UNET,
    "unet": UNET,
    "unet_plus_plus": UNET_PLUS_PLUS
}

# Import manager after constants are defined
from .manager import VoiceEnhancementModelManager

__all__ = [
    'MODIFIED_UNET',
    'UNET',
    'UNET_PLUS_PLUS',
    'MODEL_MAPPING',
    'VoiceEnhancementModelManager'
] 