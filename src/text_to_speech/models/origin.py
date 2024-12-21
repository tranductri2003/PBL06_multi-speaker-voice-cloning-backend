import torch
import torch.nn as nn
from text_to_speech.modules.text_encoder import TextEncoder
from text_to_speech.modules.mel_decoder import MelDecoder
from text_to_speech.modules.fusions import Projection

class TNTModel(nn.Module):
    def __init__(self, d_model=256, num_heads=8, device=None):
        super(TNTModel, self).__init__()
        self.encoder = TextEncoder(d_model=d_model, num_heads=num_heads, num_layers=3)
        self.projection = Projection(d_model)
        self.decoder = MelDecoder(d_model=d_model, num_heads=num_heads, num_layers=3, mel_dim=80)
        self.device = device
        self.to(self.device)
        
    def forward(self, text, pos_text, input_mel, pos_mel, encoded_speech):
        decoder_len = input_mel.size(1)
        # input_len = text.size(1)
        
        if self.training:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(decoder_len, device=input_mel.device)
            # memory_mask = self._generate_subsequent_mask(decoder_len, input_len).to(input_mel.device)
            memory_mask = None
            tgt_key_padding_mask = pos_mel.eq(0)
            src_key_padding_mask = memory_key_padding_mask = pos_text.eq(0)
            src_mask = None
        else:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(decoder_len, device=input_mel.device)
            memory_mask = None
            src_mask, tgt_key_padding_mask, src_key_padding_mask, memory_key_padding_mask = None, None, None, None
        
        memory = self.encoder(text, pos_text, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        memory = self.projection(encoded_speech, memory)
        mel_output, postnet_ouput = self.decoder(memory, input_mel, pos_mel, memory_mask=memory_mask, memory_key_padding_mask=memory_key_padding_mask, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        return mel_output, postnet_ouput