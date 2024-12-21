import base64
import time
from io import BytesIO
from fastapi import APIRouter, UploadFile, File, Form
import io
import soundfile as sf
from text_to_speech.models import EN_TACOTRON, MEL2MAG
from text_to_speech.configs.audio_config import Text2SpeechAudioConfig
from speaker_verification.models import LSTM_SPEAKER_ENCODER
from text_to_speech.services.generate_speech import generate_speech
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/voice-cloning")


@router.post("/tacotron")
async def text2speech_model1(
    text: str = Form(...),  audio: UploadFile = File(...)
):
    start_time = time.time()

    audio = generate_speech(text = text, audio = BytesIO(await audio.read()), text_toS_speech_model = EN_TACOTRON.model, speaker_verification_model=LSTM_SPEAKER_ENCODER, mel2mag_nodel=MEL2MAG.model)
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, audio, samplerate=Text2SpeechAudioConfig.SAMPLE_RATE, format='WAV')
    audio_buffer.seek(0)
    
    base64_audio = base64.b64encode(audio_buffer.read()).decode("utf-8")
    return JSONResponse(content={"audio_base64": base64_audio})

    
