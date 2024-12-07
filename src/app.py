from fastapi import FastAPI
from speaker_verification.routers import speaker_verification_router
from voice_enhancement.routers import voice_enhancement_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Voice Processing API",
    description="API for speaker verification and voice enhancement",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add routers
app.include_router(speaker_verification_router, prefix="/api/v1", tags=["speaker-verification"])
app.include_router(voice_enhancement_router, prefix="/api/v1", tags=["voice-enhancement"])
