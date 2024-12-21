import torch.nn as nn
from text_to_speech.modules.decoder_prenet import DecoderPrenet
from text_to_speech.modules.positional_encoding import ScaledPositionalEncoding
from text_to_speech.modules.post_conv_net import PostConvNet


class MelDecoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, mel_dim, max_position_encoding=8192, dropout=0.1, padding_idx=0):
        super(MelDecoder, self).__init__()
        
        # Decoder prenet: 3 linear layers with xavier uniform initialization
        self.prenet = DecoderPrenet(input_dim=mel_dim, hidden_dim=d_model, num_layers=3)
        
        # Scaled Positional Encoding
        self.positional_encoding = ScaledPositionalEncoding(d_model, n_position=max_position_encoding, dropout=dropout)
        
        # Transformer Decoder Layers
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout, dim_feedforward=d_model)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output layer for mel spectrogram prediction
        self.mel_linear = nn.Linear(d_model, mel_dim)
        
        self.postconvnet = PostConvNet(d_model)


    def forward(self, memory, input_mel, pos_mel, memory_mask=None, memory_key_padding_mask=None, tgt_mask=None, tgt_key_padding_mask=None):
        # Prenet processing on mel spectrogram input
        mel_embeds = self.prenet(input_mel)
        
        # Positional encoding for mel spectrogram
        mel_embeds = self.positional_encoding(mel_embeds, pos_mel)
        
        # Reshape for TransformerDecoder (sequence length first)
        mel_embeds_reshaped = mel_embeds.permute(1, 0, 2)  # [seq_len, batch_size, d_model]
        memory_reshaped = memory.permute(1, 0, 2)          # [seq_len, batch_size, d_model]
        

        # Transformer decoder: Multi-head attention and feed-forward layers
        decoded_output = self.transformer_decoder(
            tgt=mel_embeds_reshaped, 
            memory=memory_reshaped,
            memory_mask=memory_mask,
            tgt_mask=tgt_mask,                      # Attention mask for target sequence
            # memory_key_padding_mask=memory_key_padding_mask,  # Key padding mask for memory (encoder outputs)
            # tgt_key_padding_mask=tgt_key_padding_mask,        # Key padding mask for target (mel spectrograms)
#             tgt_is_causal=True,
#             memory_is_causal=True
        )
        
        decoded_output = decoded_output.permute(1, 0, 2)  # Reshape back to [batch_size, seq_len, d_model]
        
        # Linear layer to predict mel spectrogram
        mel_output = self.mel_linear(decoded_output)
        postnet_ouput = self.postconvnet(mel_output.permute(0, 2, 1))
        postnet_ouput = postnet_ouput.permute(0, 2, 1)
        return mel_output, postnet_ouput