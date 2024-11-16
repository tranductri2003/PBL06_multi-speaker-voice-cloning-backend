import time
from io import BytesIO
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse


router = APIRouter()


@router.post("/unet")
async def denoise_by_unet(audio: UploadFile = File(...)):
    return

@router.post("/modified_unet")
async def denoise_by_modified_unet(audio: UploadFile = File(...)):

    return 


@router.post("/unet_plusplus")
async def denoise_by_unet_plusplus(audio: UploadFile = File(...)):

    return