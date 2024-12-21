import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_sinusoid_encoding_table(n_position, num_hidden, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / num_hidden)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(num_hidden)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


class ScaledPositionalEncoding(nn.Module):
    def __init__(self, d_model, n_position=2048, dropout=0.1, padding_idx=None):
        super(ScaledPositionalEncoding, self).__init__()
        
        self.alpha = nn.Parameter(torch.ones(1))  # Scaling factor
        
        # Get sinusoid encoding table
        self.positional_embedding = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_model, padding_idx=padding_idx),
            freeze=True
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, pos):
        pos = self.positional_embedding(pos)
        x = pos * self.alpha + x
        return self.dropout(x)