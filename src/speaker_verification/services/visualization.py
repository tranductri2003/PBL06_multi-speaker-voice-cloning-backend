import io
import base64
import numpy as np
import matplotlib.pyplot as plt
import librosa
from io import BytesIO

def create_single_embedding_plot(embedding: np.ndarray, title: str) -> bytes:
    """Create plot for single embedding vector"""
    plt.figure(figsize=(12, 6))
    x = np.arange(len(embedding))
    plt.plot(x, embedding.flatten(), '-o', markersize=4)
    plt.title(f'Embedding Vector - {title}')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.margins(x=0.01)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf.getvalue()

def create_comparison_plot(embedding1: np.ndarray, embedding2: np.ndarray) -> bytes:
    """Create comparison plot for both embedding vectors"""
    plt.figure(figsize=(12, 6))
    x = np.arange(len(embedding1))
    
    plt.plot(x, embedding1.flatten(), '-o', label='First Audio', markersize=4)
    plt.plot(x, embedding2.flatten(), '-o', label='Second Audio', markersize=4)
    
    plt.title('Visualization of Embedding Vectors Comparison')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()
    plt.margins(x=0.01)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf.getvalue()

def visualize_embeddings(embedding1: np.ndarray, embedding2: np.ndarray) -> tuple[bytes, bytes, bytes]:
    """
    Create three visualizations: two individual plots and one comparison plot
    
    Args:
        embedding1: First embedding vector (128 dimensions)
        embedding2: Second embedding vector (128 dimensions)
    
    Returns:
        Tuple of three base64 encoded PNG images (first_plot, second_plot, comparison_plot)
    """
    plot1 = create_single_embedding_plot(embedding1, "First Audio")
    plot2 = create_single_embedding_plot(embedding2, "Second Audio")
    plot_comparison = create_comparison_plot(embedding1, embedding2)
    
    return plot1, plot2, plot_comparison 

def visualize_mel_spectrogram(mel_spec: np.ndarray) -> bytes:
    """
    Visualize mel spectrogram and return as bytes
    
    Args:
        mel_spec: Mel spectrogram array
    Returns:
        bytes of PNG image
    """
    print(f"Mel spec shape: {mel_spec.shape}")
    print(f"Mel spec range: [{mel_spec.min()}, {mel_spec.max()}]")
    print(f"Mel spec dtype: {mel_spec.dtype}")
    
    # Create plot
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        mel_spec,
        y_axis='mel',
        x_axis='time',
        sr=16000,
        hop_length=160
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    
    # Save to buffer and return bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf.getvalue() 