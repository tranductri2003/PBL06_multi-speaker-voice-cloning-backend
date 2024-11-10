from fastapi import FastAPI
from speaker_verification.routers import router as speaker_verification_router
# from voice_cloning import router as voice_cloning_router
# from voice_enhancement import router as voice_enhancement_router

app = FastAPI(title="AI Services")

# Include each sub-app router
app.include_router(speaker_verification_router, prefix="/api/speaker_verification")
# app.include_router(voice_cloning_router, prefix="/api/voice_cloning")
# app.include_router(voice_enhancement_router, prefix="/api/voice_enhancement")
