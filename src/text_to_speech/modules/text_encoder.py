import torch.nn as nn

from text_to_speech.modules.encoder_prenet import EncoderPrenet
from text_to_speech.modules.positional_encoding import ScaledPositionalEncoding

class TextEncoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, embedding_size=91,max_position_encoding=4096*2, dropout=0.1):
        super(TextEncoder, self).__init__()
        # Encoder Prenet
        self.encoder_prenet = EncoderPrenet(embedding_size, d_model, kernel_size=5)
        
        
        # Scaled Positional Encoding
        self.positional_encoding = ScaledPositionalEncoding(d_model, n_position=max_position_encoding, dropout=dropout, padding_idx=0)
        
        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout, dim_feedforward=d_model, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        
    def forward(self, text, pos_text, src_mask=None, src_key_padding_mask=None):
        text = self.encoder_prenet(text)
        
        # Positional encoding
        text = self.positional_encoding(text, pos_text)

        # Pass through transformer encoder
        encoded_text = self.transformer_encoder(text)
        
        return encoded_text