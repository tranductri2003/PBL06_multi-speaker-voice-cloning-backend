from typing import Dict, Optional
from core.utils.model import TensorFlowModel
from . import MODEL_MAPPING

class VoiceEnhancementModelManager:
    """Manager for voice enhancement models"""
    
    def __init__(self):
        self._models: Dict[str, TensorFlowModel] = {}
        # Pre-load all models
        for model_type, model in MODEL_MAPPING.items():
            self._models[model_type] = model
    
    def get_model(self, model_type: str) -> TensorFlowModel:
        """Get model by type"""
        if model_type not in self._models:
            raise ValueError(f"Model type '{model_type}' not supported. Available types: {list(self._models.keys())}")
        return self._models[model_type]
    
    def load_model(self, model_type: str) -> TensorFlowModel:
        """Load a specific model if not already loaded"""
        if model_type not in MODEL_MAPPING:
            raise ValueError(f"Model type '{model_type}' not supported. Available types: {list(MODEL_MAPPING.keys())}")
            
        if model_type not in self._models:
            self._models[model_type] = MODEL_MAPPING[model_type]
            
        return self._models[model_type]
    
    def load_all_models(self) -> None:
        """Load all available models"""
        for model_type in MODEL_MAPPING:
            self.load_model(model_type)
    
    def unload_model(self, model_type: str) -> None:
        """Unload a specific model from memory"""
        if model_type in self._models:
            del self._models[model_type]
    
    def predict(self, model_type: str, *args, **kwargs):
        """Make prediction using specified model"""
        model = self.get_model(model_type)
        return model.predict(*args, **kwargs)
    
    def get_loaded_models(self) -> Dict[str, bool]:
        """Get dictionary of loaded models"""
        return {
            model_type: model_type in self._models
            for model_type in MODEL_MAPPING.keys()
        }
    
    @classmethod
    def list_available_models(cls) -> Dict[str, bool]:
        """List all available models"""
        return {
            model_type: True for model_type in MODEL_MAPPING.keys()
        } 