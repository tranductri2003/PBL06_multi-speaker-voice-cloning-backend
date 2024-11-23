import base64
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from voice_enhancement.services.denoise import predict_denoised_audio
from voice_enhancement.models import VoiceEnhancementModelManager

router = APIRouter(prefix="/voice-enhancement")
model_manager = VoiceEnhancementModelManager()

@router.post("/denoise")
async def denoise_audio(
    audio: UploadFile = File(...),
    model_name: str = Form(...),
):
    """
    Endpoint for denoising audio files
    
    Args:
        audio: Audio file to denoise
        model_name: Name of the model to use (modified_unet, unet, or unet_plus_plus)
    
    Returns:
        JSON response containing denoised audio and spectrograms in base64 format
    """
    try:
        # Validate model name and get model
        model = model_manager.get_model(model_name)
        
        # Read audio file
        audio_bytes = await audio.read()

        # Call prediction function
        result = predict_denoised_audio(
            audio_bytes=audio_bytes,
            model=model
        )
        
        # Prepare response
        response = {
            "audio_base64": base64.b64encode(result.output_audio.read()).decode("utf-8"),
            "model_name": model_name,
            "spectrogram_voice_noise": base64.b64encode(result.spectrogram_voice_noise).decode("utf-8"),
            "spectrogram_predicted_noise": base64.b64encode(result.spectrogram_predicted_noise).decode("utf-8"),
            "spectrogram_predicted_voice": base64.b64encode(result.spectrogram_predicted_voice).decode("utf-8"),
            "wave_voice_noise": base64.b64encode(result.wave_voice_noise).decode("utf-8"),
            "wave_predicted_noise": base64.b64encode(result.wave_predicted_noise).decode("utf-8"),
            "wave_predicted_voice": base64.b64encode(result.wave_predicted_voice).decode("utf-8")
        }
        
        return JSONResponse(content=response)

    except ValueError as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=400  # Bad Request for invalid model name
        )
    except Exception as e:
        return JSONResponse(
            content={"error": f"Internal server error: {str(e)}"},
            status_code=500
        )
