"""
    A main file contains all APIs and runs server
"""

import os
from io import BytesIO
from typing import Literal
from dotenv import load_dotenv
import torch
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from utils.speaker_encoder.models.LstmEncoder import SpeakerLstmEncoder
from utils.speaker_encoder.models.TransformerEncoder import SpeakerTransformerEncoder

from actions.compare_audio import preprocess_audio, compare_embeddings

app = FastAPI()

load_dotenv()
# Load pre-trained SpeakerEncoder model
device = os.getenv('device', 'cpu')
model = SpeakerTransformerEncoder(device=device)

model_path = os.environ.get('MODEL_PATH')
ckpt = torch.load(model_path, weights_only=False)
if ckpt:
    print(f'loading ckpt {model_path}')
    model.load_state_dict(ckpt['model_state_dict'])
model.eval() # Put the model in evaluation mode


@app.post("/compare-speakers/")
async def compare_speakers(audio1: UploadFile = File(...), audio2: UploadFile = File(...), 
                           model_file: UploadFile = None, model_type: Literal["transformer", "lstm"] = Form("transformer"),
                           num_iter: int = Form(33), sequence_length_in_ms: int = Form(8)):
    """
    API for testing speaker verification models and calculating the cosine similarity between audio1 and audio2
    Args:
        audio1 (UploadFile, optional): The first Audio for calculating the cosine similarity. Defaults to File(...).
        audio2 (UploadFile, optional): The second Audio for calculating the cosine similarity. Defaults to File(...).
        model_file (UploadFile, optional): A model file for testing or calculating the cosine similarity. Defaults to None.
        model_type (Literal[transformer, lstm], optional): Type of model_file. Defaults to Form("transformer").
        num_iter (int, optional): _description_. Defaults to Form(33).

    Returns:
        _type_: _description_
    """
    used_model = model
    curr_model_file = 'transformer'
    sequence_length = int(sequence_length_in_ms / 1000 * 16000)
    if model_file is not None:
        if model_type == "transformer":
            used_model = SpeakerTransformerEncoder(device=device)
        elif model_type == "lstm":
            curr_model_file = 'lstm'
            used_model = SpeakerLstmEncoder(device=device)
        else:
            return JSONResponse(content={"message": "Invalid model type. Choose either 'lstm' or 'transformer'."}, status_code=400)

        curr_ckpt = torch.load(BytesIO(await model_file.read()), weights_only=False)
        if curr_ckpt:
            try:
                used_model.load_state_dict(curr_ckpt['model_state_dict'])
            except Exception:
                return JSONResponse(content={"message": "Error loading model. Please make sure the model file is in the correct format and compatible with the chosen model type."}, 
                                    status_code=400)
    used_model.to(device)

    # Preprocess both audio files
    audio1 = preprocess_audio(BytesIO(await audio1.read()), seq_len=sequence_length, num_iter=num_iter)
    audio2 = preprocess_audio(BytesIO(await audio2.read()), seq_len=sequence_length, num_iter=num_iter)

    # Generate embeddings for both audio files
    with torch.no_grad():
        embedding1 = model(audio1)
        embedding2 = model(audio2)

    # Compute cosine similarity between the two embeddings
    similarity = compare_embeddings(torch.mean(embedding1, dim=0, keepdim=True), torch.mean(embedding2, dim=0, keepdim=True))
    return JSONResponse(content={"similarity_score": similarity, "model_type": curr_model_file, "num_iter":num_iter, "msg": "Successfull"}, status_code=200)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
