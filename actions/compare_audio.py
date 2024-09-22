"""
    Actions for Compare Speakers
"""

from io import BytesIO
import torch
from torch.nn.functional import cosine_similarity
import numpy as np
from utils.objects import Utterance


def preprocess_audio(file: BytesIO, seq_len=128, num_iter=33) -> torch.Tensor:
    """_summary_

    Args:
        file (BytesIO): audio file
        seq_len (int, optional): legth of audio which you want to use to generate mel. Defaults to 128.
        num_iter (int, optional): Number of samples for calculating the average. Defaults to 33.

    Returns:
        torch.Tensor: A PyTorch tensor contains num_iter samples, each with a length of seq_len
    """
    uttn = Utterance(raw_file=file)
    return torch.tensor(
        np.array([uttn.random_mel_segment(seq_len=seq_len) for _ in (num_iter)])
    ).transpose(1, 2)


def compare_embeddings(embed1: torch.Tensor, embed2: torch.Tensor):
    """_summary_

    Args:
        embed1 (torch.Tensor): Featured Vector of Audio 1
        embed2 (torch.Tensor): Featured Vector of Audio 2

    Returns:
        Number: cosine similarity value
    """
    similarity = cosine_similarity(embed1, embed2)
    return similarity.item()
