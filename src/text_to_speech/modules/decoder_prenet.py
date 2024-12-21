import torch.nn as nn

class DecoderPrenet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=3):
        super(DecoderPrenet, self).__init__()
        layers = []
        for _ in range(num_layers):
            linear_layer = nn.Linear(input_dim, hidden_dim)
            nn.init.xavier_uniform_(linear_layer.weight)  # Xavier initialization
            layers.append(linear_layer)
            layers.append(nn.ReLU())
            input_dim = hidden_dim  # Update input dimension for next layer
        self.prenet = nn.Sequential(*layers)

    def forward(self, x):
        return self.prenet(x)