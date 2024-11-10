from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from torch.nn.utils import clip_grad_norm_
from scipy.optimize import brentq
import torch
import torch.nn as nn
import numpy as np


class LstmSpeakerEncoder(nn.Module):
    def __init__(
        self,
        input_size=80,
        hidden_size=256,
        num_layers=3,
        bidirectional=True,
        device="cpu",
        loss_device="cpu",
    ):
        super().__init__()
        self.loss_device = loss_device

        # Network defition
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        ).to(device)
        self.linear = nn.Linear(
            in_features=hidden_size if bidirectional else hidden_size * 2,
            out_features=256,
        ).to(device)
        self.relu = torch.nn.ReLU().to(device)

        # Cosine similarity scaling (with fixed initial parameter values)
        self.similarity_weight = nn.Parameter(torch.tensor([10.0], device=loss_device))
        self.similarity_bias = nn.Parameter(torch.tensor([-5.0], device=loss_device))

        # Loss
        self.loss_fn = nn.CrossEntropyLoss().to(loss_device)

    def do_gradient_ops(self):
        # Gradient scale
        self.similarity_weight.grad *= 0.01
        self.similarity_bias.grad *= 0.01

        # Gradient clipping
        clip_grad_norm_(self.parameters(), 3, norm_type=2)

    def forward(self, utterances, hidden_init=None):
        """
        Computes the embeddings of a batch of utterance spectrograms.
        """

        out, (hidden, cell) = self.lstm(utterances, hidden_init)

        # We take only the hidden state of the last layer
        embeds_raw = self.relu(self.linear(hidden[-1]))

        # L2-normalize it
        embeds = embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)

        return embeds

    def similarity_matrix(self, embeds):
        """
        Computes the similarity matrix according the section 2.1 of GE2E.
        """
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]

        # Inclusive centroids (1 per speaker). Cloning is needed for reverse differentiation
        centroids_incl = torch.mean(embeds, dim=1, keepdim=True)
        centroids_incl = centroids_incl.clone() / torch.norm(
            centroids_incl, dim=2, keepdim=True
        )

        # Exclusive centroids (1 per utterance)
        centroids_excl = torch.sum(embeds, dim=1, keepdim=True) - embeds
        centroids_excl /= utterances_per_speaker - 1
        centroids_excl = centroids_excl.clone() / torch.norm(
            centroids_excl, dim=2, keepdim=True
        )

        sim_matrix = torch.zeros(
            speakers_per_batch, utterances_per_speaker, speakers_per_batch
        ).to(self.loss_device)
        mask_matrix = 1 - np.eye(speakers_per_batch, dtype=np.int64)
        for j in range(speakers_per_batch):
            mask = np.where(mask_matrix[j])[0]
            sim_matrix[mask, :, j] = (embeds[mask] * centroids_incl[j]).sum(dim=2)
            sim_matrix[j, :, j] = (embeds[j] * centroids_excl[j]).sum(dim=1)

        ## Even more vectorized version (slower maybe because of transpose)
        sim_matrix = sim_matrix * self.similarity_weight + self.similarity_bias
        return sim_matrix

    def loss(self, embeds):
        """
        Computes the softmax loss of GE2E.
        """
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]

        # Loss
        sim_matrix = self.similarity_matrix(embeds)
        sim_matrix = sim_matrix.reshape(
            (speakers_per_batch * utterances_per_speaker, speakers_per_batch)
        )
        ground_truth = np.repeat(np.arange(speakers_per_batch), utterances_per_speaker)
        target = torch.from_numpy(ground_truth).long().to(self.loss_device)
        loss = self.loss_fn(sim_matrix, target)

        # EER (not backpropagated)
        with torch.no_grad():
            inv_argmax = lambda i: np.eye(1, speakers_per_batch, i, dtype=np.int64)[0]
            labels = np.array([inv_argmax(i) for i in ground_truth])
            preds = sim_matrix.detach().cpu().numpy()

            fpr, tpr, thresholds = roc_curve(labels.flatten(), preds.flatten())
            eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)

        return loss, eer

    def __str__(self):
        return "LstmSpeakerEncoder"
