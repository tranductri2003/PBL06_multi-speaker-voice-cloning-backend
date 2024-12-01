import torch
from torch.nn.functional import cosine_similarity
from speaker_verification.services.data_preprocess import preprocess_audio
from speaker_verification.services.visualization import visualize_mel_spectrogram


def calculate_cosine_similarity(model, audio1, audio2):
    # Get mel spectrograms, cleaned audio and visualizations
    mel_spec1, clean_audio1, mel_viz1 = preprocess_audio(audio1)
    mel_spec2, clean_audio2, mel_viz2 = preprocess_audio(audio2)

    with torch.no_grad():
        # Get embeddings from mel spectrograms
        embedding1 = model(mel_spec1)
        embedding2 = model(mel_spec2)

        # Calculate mean embeddings
        mean_embedding1 = torch.mean(embedding1, dim=0)
        mean_embedding2 = torch.mean(embedding2, dim=0)
        
        # Calculate similarity using unsqueezed tensors
        similarity = cosine_similarity(
            mean_embedding1.unsqueeze(0),
            mean_embedding2.unsqueeze(0)
        )
    
    # Visualize mel spectrograms
    mel_viz1_img = visualize_mel_spectrogram(mel_viz1)
    mel_viz2_img = visualize_mel_spectrogram(mel_viz2)
    
    return (similarity.item(), mel_viz1_img, mel_viz2_img), (clean_audio1, clean_audio2)
