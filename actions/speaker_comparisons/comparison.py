import torch
from settings import SPEAKER_TRANSFORMER_ENCODER
from torch.nn.functional import cosine_similarity
from actions.speaker_comparisons.processor import preprocess_audio


def calculate_cosine_similarity(model, audio1, audio2):
    processed_audio1 = preprocess_audio(audio1)
    processed_audio2 = preprocess_audio(audio2)

    with torch.no_grad():
        embedding1 = model(processed_audio1)
        embedding2 = model(processed_audio2)

    similarity = cosine_similarity(
        torch.mean(embedding1, dim=0, keepdim=True),
        torch.mean(embedding2, dim=0, keepdim=True),
    )
    return similarity.item()
