import time
import numpy as np
from io import BytesIO
from fastapi import APIRouter, UploadFile, File, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.responses import Response
import base64
from speech_enhancement.services.denoise import predict_denoised_audio
import soundfile as sf

router = APIRouter()

@router.post("/denoise/{model_name}")
async def denoise_by_modified_unet(
    audio: UploadFile = File(...),
    model_name: str = None,
):
    try:
        audio_bytes = await audio.read()

        # Call prediction function
        output_audio, spectrogram_m_amp_db, spectrogram_inv_sca_X_pred, spectrogram_X_denoise, wave_m_amp_db, wave_inv_sca_X_pred, wave_X_denoise = predict_denoised_audio(
            audio_bytes, model_name
        )

            # Return denoised audio
            # return StreamingResponse(
            #     output_audio,
            #     media_type="audio/wav",
            #     headers={"Content-Disposition": f"attachment; filename=denoised_{audio.filename}"}
            # )
            # Return spectrograms as base64-encoded images
        audio_base64 = base64.b64encode(output_audio.read()).decode("utf-8")

        spectrograms = {
            
        }
        response = {
            "audio_base64": audio_base64,
            "model_name": model_name,
            "spectrogram_voice_noise": base64.b64encode(spectrogram_m_amp_db).decode("utf-8"),
            "spectrogram_predicted_noise": base64.b64encode(spectrogram_inv_sca_X_pred).decode("utf-8"),
            "spectrogram_predicted_voice": base64.b64encode(spectrogram_X_denoise).decode("utf-8"),
            "wave_voice_noise": base64.b64encode(wave_m_amp_db).decode("utf-8"),
            "wave_predicted_noise": base64.b64encode(wave_inv_sca_X_pred).decode("utf-8"),
            "wave_predicted_voice": base64.b64encode(wave_X_denoise).decode("utf-8"),
        }
        return JSONResponse(content=response)

    except Exception as e:
        print("Error:", str(e))
        return {"error": str(e)}
