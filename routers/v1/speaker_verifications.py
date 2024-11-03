import time
import os
from io import BytesIO
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import torch

from settings import SPEAKER_TRANSFORMER_ENCODER, SPEAKER_LSTM_ENCODER
from actions.speaker_comparisons import (
    calculate_cosine_similarity,
    test_transformer_speaker_verification,
    test_lstm_speaker_verification
)

router = APIRouter()


@router.post("/transformer/compare_speakers")
async def transfromer_compare_speakers(
    first_audio: UploadFile = File(...), second_audio: UploadFile = File(...)
):
    start_time = time.time()
    similarity = calculate_cosine_similarity(
        SPEAKER_TRANSFORMER_ENCODER,
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


@router.get("/transformer/export_test")
async def transformer_compare_speakers():
    start_time = time.time()
    test_result = test_transformer_speaker_verification()
    return JSONResponse(
        content={
            "test_result": test_result,
            "duration": time.time() - start_time,
            "msg": "Successfull",
        },
        status_code=200,
    )


@router.post("/lstm/compare_speakers")
async def lstm_compare_speakers(
    first_audio: UploadFile = File(...), second_audio: UploadFile = File(...)
):
    start_time = time.time()
    similarity = calculate_cosine_similarity(
        SPEAKER_LSTM_ENCODER,
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
    
@router.get("/lstm/export_test")
async def lstm_compare_speakers():
    start_time = time.time()
    test_result = test_lstm_speaker_verification()
    return JSONResponse(
        content={
            "test_result": test_result,
            "duration": time.time() - start_time,
            "msg": "Successfull",
        },
        status_code=200,
    )