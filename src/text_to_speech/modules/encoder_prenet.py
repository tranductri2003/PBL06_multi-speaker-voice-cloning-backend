import torch.nn as nn

class EncoderPrenet(nn.Module):
    def __init__(self, input_dim, conv_channels, kernel_size, num_layers=3):
        super(EncoderPrenet, self).__init__()
        layers = []
        for _ in range(num_layers):
            conv_layer = nn.Conv1d(input_dim, conv_channels, kernel_size, padding=kernel_size//2)
            nn.init.xavier_uniform_(conv_layer.weight)  # Xavier initialization
            layers.append(conv_layer)
            layers.append(nn.ReLU())
            input_dim = conv_channels  # Update input dimension for next layer
        self.prenet = nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.prenet(x)
        return x.transpose(1, 2)
