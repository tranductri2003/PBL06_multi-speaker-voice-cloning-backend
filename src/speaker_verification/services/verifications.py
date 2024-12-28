import torch
from torch.nn.functional import cosine_similarity
from speaker_verification.services.data_preprocess import preprocess_audio
from speaker_verification.services.visualization import visualize_mel_spectrogram
import numpy as np
from speaker_verification.configs.audio_config import LstmSpeakerEncoderAudioConfig, TransformerSpeakerEncoderAudioConfig
from core.utils.objects.utterance import Utterance
from core.utils.processors.audio_processor import AudioPreprocessor

def embed_frames_batch(frames_batch, model, device):
    if model is None:
        raise Exception("Model was not loaded. Call load_model() before inference.")

    frames = torch.from_numpy(frames_batch).to(device)
    embed = model.forward(frames).detach().cpu().numpy()
    return embed

def embed_utterance(model,utterance, using_partials=False, device="cpu"):
    # Process the entire utterance if not using partials
    if not using_partials:
        frames = utterance.mel_in_db()
        frames_batch=frames[None, ...].transpose(0, 2, 1)
        embed = embed_frames_batch(model=model,frames_batch=frames_batch,device=device)
        return embed

    # Compute where to split the utterance into partials and pad if necessary
    frames_list = utterance.split_mel_into_frames()
    frames_batch = np.stack(frames_list, axis=2)
    # Pass to the model
    partial_embeds = embed_frames_batch(model=model, frames_batch=frames_batch, device=device)

    # Compute the utterance embedding from the partial embeddings
    raw_embed = partial_embeds.cpu().mean(axis=0, keepdim=True)
    embed = raw_embed / np.linalg.norm(raw_embed, 2)

    return embed

def calculate_cosine_similarity(model, audio1, audio2, model_type="lstm"):
    config = LstmSpeakerEncoderAudioConfig if model_type == "lstm" else TransformerSpeakerEncoderAudioConfig
    # Get mel spectrograms, cleaned audio and visualizations
    mel_spec1, clean_audio1, mel_viz1 = preprocess_audio(audio1, model_type=model_type)
    mel_spec2, clean_audio2, mel_viz2 = preprocess_audio(audio2, model_type=model_type)
    
    clean_uttn1 = Utterance(
            raw_file=clean_audio1,
            processor=AudioPreprocessor(config=config)
        )
    
    clean_uttn2 = Utterance(
            raw_file=clean_audio2,
            processor=AudioPreprocessor(config=config)
        )

    with torch.no_grad():
        # Get embeddings from mel spectrograms
        embedding1 = embed_utterance(model, clean_uttn1)
        embedding2 = embed_utterance(model, clean_uttn2)

        # Calculate similarity using unsqueezed tensors
        similarity = cosine_similarity(
            torch.tensor(embedding1),
            torch.tensor(embedding2)
        )
    
    # Visualize mel spectrograms
    mel_viz1_img = visualize_mel_spectrogram(mel_viz1)
    mel_viz2_img = visualize_mel_spectrogram(mel_viz2)
    
    return (similarity.item(), mel_viz1_img, mel_viz2_img), (clean_audio1, clean_audio2)
