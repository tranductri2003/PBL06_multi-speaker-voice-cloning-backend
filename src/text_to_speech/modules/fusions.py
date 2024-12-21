import torch
import torch.nn as nn

class Projection(nn.Module):
    def __init__(self, d_model):
        super(Projection, self).__init__()
        self.fc = nn.Linear(d_model * 2, d_model)  # Concatenates Text + Audio encodings

    def forward(self, encoded_speech, memory):

        duplicated_encoded_speech = encoded_speech.repeat(1, memory.shape[1], 1)


        # Concatenate along the feature dimension
        concat_enc = torch.cat((duplicated_encoded_speech, memory), dim=-1)

        # Pass through fully connected layer
        return self.fc(concat_enc)