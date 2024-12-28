import torch
import torch.nn as nn

from text_to_speech.modules.batchnorm import BatchNormConv
from text_to_speech.modules.highwaynet import HighwayNetwork


class CBHG(nn.Module):
    def __init__(self, K, in_channels, channels, proj_channels, num_highways):
        super().__init__()

        self._to_flatten = []

        self.bank_kernels = [i for i in range(1, K + 1)]
        self.conv1d_bank = nn.ModuleList()
        for k in self.bank_kernels:
            conv = BatchNormConv(in_channels, channels, k)
            self.conv1d_bank.append(conv)

        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

        self.conv_project1 = BatchNormConv(len(self.bank_kernels) * channels, proj_channels[0], 3)
        self.conv_project2 = BatchNormConv(proj_channels[0], proj_channels[1], 3, relu=False)

        if proj_channels[-1] != channels:
            self.highway_mismatch = True
            self.pre_highway = nn.Linear(proj_channels[-1], channels, bias=False)
        else:
            self.highway_mismatch = False

        self.highways = nn.ModuleList()
        for i in range(num_highways):
            hn = HighwayNetwork(channels)
            self.highways.append(hn)

        self.rnn = nn.GRU(channels, channels // 2, batch_first=True, bidirectional=True)
        self._to_flatten.append(self.rnn)

        self._flatten_parameters()

    def forward(self, x):
        self._flatten_parameters()
        
        residual = x
        seq_len = x.size(-1)
        conv_bank = []

        for conv in self.conv1d_bank:
            c = conv(x)
            conv_bank.append(c[:, :, :seq_len])
            
        conv_bank = torch.cat(conv_bank, dim=1)
        x = self.maxpool(conv_bank)[:, :, :seq_len]
        x = self.conv_project1(x)
        x = self.conv_project2(x)
        x = x + residual
        x = x.transpose(1, 2)
        
        if self.highway_mismatch is True:
            x = self.pre_highway(x)
            
        for h in self.highways: 
            x = h(x)
        
        x, _ = self.rnn(x)
        return x

    def _flatten_parameters(self):
        [m.flatten_parameters() for m in self._to_flatten]