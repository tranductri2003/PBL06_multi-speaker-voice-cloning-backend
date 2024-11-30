import time
import base64
from enum import Enum
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from voice_enhancement.services.denoise import predict_denoised_audio
from voice_enhancement.models import VoiceEnhancementModelManager

router = APIRouter(prefix="/voice-enhancement")
model_manager = VoiceEnhancementModelManager()

class ModelType(str, Enum):
    MODIFIED_UNET = "modified_unet"
    UNET = "unet"
    UNET_PLUS_PLUS = "unet_plus_plus"
    CNN50 = "cnn50"
    CNN100 = "cnn100"

@router.post("/denoise")
async def denoise_audio(
    audio: UploadFile = File(...),
    model_type: ModelType = Form(...),
):
    """
    Endpoint for denoising audio files
    
    Args:
        audio: Audio file to denoise
        model_type: Type of model to use (modified_unet, unet, or unet_plus_plus)
    
    Returns:
        JSON response containing denoised audio and spectrograms in base64 format
    """
    try:
        start_time = time.time()
        
        model = model_manager.get_model(model_type)
        
        audio_bytes = await audio.read()

        result = predict_denoised_audio(
            audio_bytes=audio_bytes,
            model=model
        )
        
        # Prepare response
        response = {
            "audio_base64": base64.b64encode(result.output_audio.read()).decode("utf-8"),
            "model_type": model_type,
            "spectrogram_voice_noise": base64.b64encode(result.spectrogram_voice_noise).decode("utf-8"),
            "spectrogram_predicted_noise": base64.b64encode(result.spectrogram_predicted_noise).decode("utf-8"),
            "spectrogram_predicted_voice": base64.b64encode(result.spectrogram_predicted_voice).decode("utf-8"),
            "wave_voice_noise": base64.b64encode(result.wave_voice_noise).decode("utf-8"),
            "wave_predicted_noise": base64.b64encode(result.wave_predicted_noise).decode("utf-8"),
            "wave_predicted_voice": base64.b64encode(result.wave_predicted_voice).decode("utf-8"),
            "duration": time.time() - start_time,
            "msg": "successful"
        }
        
        return JSONResponse(content=response)

    except ValueError as e:
        return JSONResponse(
            content={
                "error": str(e),
                "msg": "failed",
                "duration": time.time() - start_time
            },
            status_code=400  
        )
    except Exception as e:
        return JSONResponse(
            content={
                "error": f"Internal server error: {str(e)}",
                "msg": "failed",
                "duration": time.time() - start_time 
            },
            status_code=500
        )
