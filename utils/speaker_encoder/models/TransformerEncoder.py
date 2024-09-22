"""
    Speaker Verification Transformer Encoder Model
"""

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import numpy as np

class SpeakerTransformerEncoder(nn.Module):
    """
    A neural network model for encoding speaker utterances using a Transformer encoder.

    This model processes Mel spectrograms of audio utterances to produce embeddings
    suitable for speaker recognition tasks. It includes methods for calculating
    similarity matrices and computing loss based on the GE2E (Generalized End-to-End) approach.

    Attributes:
        loss_device (str): The device to use for loss computation (e.g., 'cpu' or 'cuda').
        transformer_encoder (nn.TransformerEncoder): Transformer encoder for processing input utterances.
        linear (nn.Linear): Linear layer for projecting outputs to embedding space.
        relu (torch.nn.ReLU): ReLU activation function.
        similarity_weight (torch.nn.Parameter): Weight for scaling cosine similarity.
        similarity_bias (torch.nn.Parameter): Bias for adjusting similarity.
        loss_fn (nn.CrossEntropyLoss): Loss function used for training.
    """

    def __init__(self, input_size=80, hidden_size=256, num_layers=3, num_heads=8, device='cpu', loss_device='cpu'):
        """
        Initialize the SpeakerTransformerEncoder.

        Args:
            input_size (int, optional): Number of input features (e.g., Mel bands). Defaults to 80.
            hidden_size (int, optional): Number of features in the feedforward layer of the Transformer. Defaults to 256.
            num_layers (int, optional): Number of Transformer encoder layers. Defaults to 3.
            num_heads (int, optional): Number of attention heads in the Transformer. Defaults to 8.
            device (str, optional): Device for the model's parameters (e.g., 'cpu' or 'cuda'). Defaults to 'cpu'.
            loss_device (str, optional): Device for loss computation. Defaults to 'cpu'.
        """
        super().__init__()
        self.loss_device = loss_device

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=num_heads,
            dim_feedforward=hidden_size,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers).to(device)

        self.linear = nn.Linear(in_features=input_size, out_features=256).to(device)
        self.relu = nn.ReLU().to(device)

        # Cosine similarity scaling (with fixed initial parameter values)
        self.similarity_weight = nn.Parameter(torch.tensor([10.])).to(loss_device)
        self.similarity_bias = nn.Parameter(torch.tensor([-5.])).to(loss_device)

        # Loss
        self.loss_fn = nn.CrossEntropyLoss().to(loss_device)

    def do_gradient_ops(self):
        """
        Apply gradient operations including scaling and clipping.

        This method scales the gradients of similarity parameters and clips the gradients
        of all parameters to prevent exploding gradients.
        """
        # Gradient scale
        self.similarity_weight.grad *= 0.01
        self.similarity_bias.grad *= 0.01

        # Gradient clipping
        clip_grad_norm_(self.parameters(), 3, norm_type=2)

    def forward(self, utterances):
        """
        Compute the embeddings of a batch of utterance spectrograms.

        Args:
            utterances (torch.Tensor): Batch of Mel spectrograms of shape (batch_size, n_frames, n_channels).
            hidden_init (torch.Tensor, optional): Not used in the Transformer version.

        Returns:
            torch.Tensor: Embeddings of shape (batch_size, embedding_size).
        """
        # Pass the input through the Transformer Encoder
        out = self.transformer_encoder(utterances)

        # Mean pooling over time steps
        embeds_raw = self.relu(self.linear(out.mean(dim=1)))

        # L2-normalize the embeddings
        embeds = embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)

        return embeds

    def similarity_matrix(self, embeds):
        """
        Compute the similarity matrix based on GE2E methodology.

        Args:
            embeds (torch.Tensor): Embeddings of shape (speakers_per_batch, utterances_per_speaker, embedding_size).

        Returns:
            torch.Tensor: Similarity matrix of shape (speakers_per_batch, utterances_per_speaker, speakers_per_batch).
        """
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]

        # Inclusive centroids (1 per speaker)
        centroids_incl = torch.mean(embeds, dim=1, keepdim=True)
        centroids_incl = centroids_incl.clone() / torch.norm(centroids_incl, dim=2, keepdim=True)

        # Exclusive centroids (1 per utterance)
        centroids_excl = (torch.sum(embeds, dim=1, keepdim=True) - embeds)
        centroids_excl /= (utterances_per_speaker - 1)
        centroids_excl = centroids_excl.clone() / torch.norm(centroids_excl, dim=2, keepdim=True)

        # Initialize similarity matrix
        sim_matrix = torch.zeros(speakers_per_batch, utterances_per_speaker,
                                 speakers_per_batch).to(self.loss_device)
        mask_matrix = 1 - np.eye(speakers_per_batch, dtype=np.int64)
        for j in range(speakers_per_batch):
            mask = np.where(mask_matrix[j])[0]
            sim_matrix[mask, :, j] = (embeds[mask] * centroids_incl[j]).sum(dim=2)
            sim_matrix[j, :, j] = (embeds[j] * centroids_excl[j]).sum(dim=1)

        # Apply similarity scaling and bias
        sim_matrix = sim_matrix * self.similarity_weight + self.similarity_bias
        return sim_matrix

    def loss(self, embeds):
        """
        Compute the softmax loss based on the similarity matrix.

        Args:
            embeds (torch.Tensor): Embeddings of shape (speakers_per_batch, utterances_per_speaker, embedding_size).

        Returns:
            tuple: A tuple containing the loss (torch.Tensor) and the equal error rate (EER) (float).
        """

        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]

        # Compute similarity matrix
        sim_matrix = self.similarity_matrix(embeds)
        sim_matrix = sim_matrix.reshape((speakers_per_batch * utterances_per_speaker,
                                         speakers_per_batch))
        ground_truth = np.repeat(np.arange(speakers_per_batch), utterances_per_speaker)
        target = torch.from_numpy(ground_truth).long().to(self.loss_device)
        loss = self.loss_fn(sim_matrix, target)

        # Compute EER (not backpropagated)
        with torch.no_grad():
            inv_argmax = lambda i: np.eye(1, speakers_per_batch, i, dtype=np.int64)[0]
            labels = np.array([inv_argmax(i) for i in ground_truth])
            preds = sim_matrix.detach().cpu().numpy()

            # Calculate EER
            fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

        return loss, eer
