import torch
import torch.nn as nn
class PostConvNet(nn.Module):
    """
    Post Convolutional Network (mel --> mel)
    """
    def __init__(self, num_hidden, num_mels=80, outputs_per_step=1, dropout_prob=0.1, device="cpu", *args, **kwargs):
        """
        :param num_hidden: Dimension of hidden layers.
        :param num_mels: Number of mel bands.
        :param outputs_per_step: Number of outputs per step.
        :param dropout_prob: Probability for dropout layers.
        """
        super(PostConvNet, self).__init__()
        
        # Define input and output channel dimensions
        self.num_mels = num_mels
        self.outputs_per_step = outputs_per_step
        self.num_hidden = num_hidden
        
        # First convolutional layer
        self.conv1 = nn.Conv1d(
            in_channels=num_mels * outputs_per_step,
            out_channels=num_hidden,
            kernel_size=5,
            padding=4
        )
        
        # Three repeated convolutional layers with batch normalization
        self.conv_list = nn.ModuleList([
            nn.Conv1d(
                in_channels=num_hidden,
                out_channels=num_hidden,
                kernel_size=5,
                padding=4
            )
            for _ in range(3)
        ])
        self.batch_norm_list = nn.ModuleList([nn.BatchNorm1d(num_hidden) for _ in range(3)])
        
        # Final convolutional layer
        self.conv2 = nn.Conv1d(
            in_channels=num_hidden,
            out_channels=num_mels * outputs_per_step,
            kernel_size=5,
            padding=4
        )
        
        # Batch normalization for the first convolution
        self.pre_batchnorm = nn.BatchNorm1d(num_hidden)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.dropout_list = nn.ModuleList([nn.Dropout(p=dropout_prob) for _ in range(3)])

    def forward(self, x, mask=None):
        """
        Forward pass of the PostConvNet.
        
        :param x: Input tensor (batch_size, num_mels * outputs_per_step, seq_len).
        :param mask: Mask (optional, not used in this implementation).
        :return: Output tensor (batch_size, num_mels * outputs_per_step, seq_len).
        """
        # Apply the first convolution, batch normalization, activation, and dropout
        x = self.conv1(x)
        x = self.pre_batchnorm(x)
        x = torch.tanh(x)
        x = self.dropout1(x[:, :, :-4])  # Slice to simulate causal behavior
        
        # Apply repeated convolutions with batch norm, activation, and dropout
        for conv, batch_norm, dropout in zip(self.conv_list, self.batch_norm_list, self.dropout_list):
            x = conv(x)
            x = batch_norm(x)
            x = torch.tanh(x)
            x = dropout(x[:, :, :-4])  # Slice to simulate causal behavior
        
        # Apply the final convolution and slicing
        x = self.conv2(x)[:, :, :-4]
        return x