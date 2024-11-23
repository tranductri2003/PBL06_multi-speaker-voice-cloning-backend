from abc import ABC, abstractmethod
import torch
import tensorflow as tf
from typing import Optional, Dict, Any, Union
from pathlib import Path

class BaseModel(ABC):
    """Abstract base class for all models"""
    def __init__(
        self,
        model_class: Optional[Any] = None,
        model_path: Optional[str] = None,
        device: str = 'cpu',
        model_type: str = 'pytorch',
        **kwargs
    ):
        self.model_class = model_class
        self.model_path = Path(model_path) if model_path else None
        self.device = device
        self.model_type = model_type.lower()
        self.model = self.load_model()

    @abstractmethod
    def predict(self, *args, **kwargs):
        """Make predictions using the model"""
        pass

    def load_model(self) -> Union[torch.nn.Module, tf.keras.Model]:
        """Load model based on type"""
        if not self.model_path or not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        try:
            if self.model_type == 'pytorch':
                return self._load_pytorch_model()
            elif self.model_type == 'tensorflow':
                return self._load_tensorflow_model()
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
        except Exception as e:
            raise RuntimeError(f"Failed to load {self.model_type} model from {self.model_path}: {str(e)}")

    def _load_pytorch_model(self) -> torch.nn.Module:
        """Load PyTorch model"""
        # Initialize model
        model = self.model_class(device=self.device)
        
        # Load weights
        checkpoint = torch.load(
            self.model_path,
            map_location=torch.device(self.device),
            weights_only=True
        )
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
        
        # Set model to evaluation mode
        model.eval()
        return model.to(self.device)

    def _load_tensorflow_model(self) -> tf.keras.Model:
        """Load TensorFlow model"""
        return tf.keras.models.load_model(str(self.model_path))


class PyTorchModel(BaseModel):
    """PyTorch specific model handler"""
    def __init__(
        self,
        model_class: torch.nn.Module,
        model_path: Optional[str] = None,
        device: str = 'cpu',
        **kwargs
    ):
        super().__init__(
            model_class=model_class,
            model_path=model_path,
            device=device,
            model_type='pytorch',
            **kwargs
        )

    def predict(self, *args, **kwargs):
        """Make predictions using PyTorch model"""
        with torch.no_grad():
            return self.model(*args, **kwargs)


class TensorFlowModel(BaseModel):
    """TensorFlow specific model handler"""
    def __init__(
        self,
        model_path: str,
        **kwargs
    ):
        super().__init__(
            model_path=model_path,
            model_type='tensorflow',
            **kwargs
        )

    def predict(self, *args, **kwargs):
        """Make predictions using TensorFlow model"""
        return self.model.predict(*args, **kwargs)
