import time
from io import BytesIO
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pathlib import Path
import io
import soundfile as sf
from text_to_speech.models import ORIGIN_TEXT_TO_SPEECH
from text_to_speech.configs.audio_config import Text2SpeechAudioConfig
from speaker_verification.models import TRANSFORMER_SPEAKER_ENCODER
from text_to_speech.services.generate_speech import generate_speech
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/voice-cloning")


@router.post("/origin")
async def text2speech_model1(
    text: str = Form(...),  audio: UploadFile = File(...)
):
    start_time = time.time()

    audio = generate_speech(text = text, audio = BytesIO(await audio.read()), text_to_speech_model = ORIGIN_TEXT_TO_SPEECH.model, speaker_verification_model=TRANSFORMER_SPEAKER_ENCODER )
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, audio, samplerate=Text2SpeechAudioConfig.SAMPLE_RATE, format='WAV')
    audio_buffer.seek(0)
    return StreamingResponse(
        audio_buffer,
        media_type="audio/wav"
    )

    
