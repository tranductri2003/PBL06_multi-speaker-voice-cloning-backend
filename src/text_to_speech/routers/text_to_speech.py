import base64
import time
from io import BytesIO
from fastapi import APIRouter, UploadFile, File, Form
import io
import soundfile as sf
from text_to_speech.models import EN_TACOTRON, MEL2MAG
from text_to_speech.configs.audio_config import Text2SpeechAudioConfig

from text_to_speech.services.generate_speech import generate_speech
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/voice-cloning")


@router.post("/tacotron")
async def text2speech_model1(
    text: str = Form(...),  audio: UploadFile = File(...), lang: str = Form(...)
):
    start_time = time.time()

    data = generate_speech(text = text, audio = BytesIO(await audio.read()), lang=lang)
    data["duration"] = time.time() - start_time
    return JSONResponse(content=data)
