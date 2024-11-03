from fastapi import FastAPI
from routers.v1 import speaker_verifications

app = FastAPI()

app.include_router(speaker_verifications.router, prefix='/api/v1')


@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI app!"}
