import time
from io import BytesIO
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse

from speaker_verification.models import TRANSFORMER_SPEAKER_ENCODER, LSTM_SPEAKER_ENCODER
from speaker_verification.services.verifications import calculate_cosine_similarity
    

router = APIRouter(prefix="/speaker-verification")

@router.post("/transformer")
async def transfromer_compare_uterrances(
    first_audio: UploadFile = File(...), second_audio: UploadFile = File(...)
):
    start_time = time.time()
    similarity = calculate_cosine_similarity(
        TRANSFORMER_SPEAKER_ENCODER.model,
        BytesIO(await first_audio.read()),
        BytesIO(await second_audio.read()),
    )
    return JSONResponse(
        content={
            "similarity_score": similarity,
            "duration": time.time() - start_time,
            "msg": "Successfull",
        },
        status_code=200,
    )


@router.post("/lstm")
async def lstm_compare_uterrances(
    first_audio: UploadFile = File(...), second_audio: UploadFile = File(...)
):
    start_time = time.time()
    similarity = calculate_cosine_similarity(
        LSTM_SPEAKER_ENCODER.model,
        BytesIO(await first_audio.read()),
        BytesIO(await second_audio.read()),
    )
    return JSONResponse(
        content={
            "similarity_score": similarity,
            "duration": time.time() - start_time,
            "msg": "Successfull",
        },
        status_code=200,
    )
