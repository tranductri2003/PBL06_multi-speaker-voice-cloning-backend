import numpy as np
import torch
import torch.nn as nn
from text_to_speech.modules.encoder import Encoder
from text_to_speech.modules.decoder import Decoder
from text_to_speech.modules.cbhg import CBHG


class Tacotron(nn.Module):
    def __init__(self, embed_dims, num_chars, encoder_dims, decoder_dims, n_mels, 
                 fft_bins, postnet_dims, encoder_K, lstm_dims, postnet_K, num_highways,
                 dropout, stop_threshold, speaker_embedding_size, *args, **kwargs):
        super().__init__()
        
        self.n_mels = n_mels
        self.lstm_dims = lstm_dims
        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims
        self.speaker_embedding_size = speaker_embedding_size
        self.encoder = Encoder(embed_dims, num_chars, encoder_dims,
                               encoder_K, num_highways, dropout)
        self.encoder_proj = nn.Linear(encoder_dims + speaker_embedding_size, decoder_dims, bias=False)
        self.decoder = Decoder(n_mels, encoder_dims, decoder_dims, lstm_dims,
                               dropout, speaker_embedding_size)
        self.postnet = CBHG(postnet_K, n_mels, postnet_dims,
                            [postnet_dims, fft_bins], num_highways)
        self.post_proj = nn.Linear(postnet_dims, fft_bins, bias=False)

        self.init_model()
        self.num_params()

        self.register_buffer("step", torch.zeros(1, dtype=torch.long))
        self.register_buffer("stop_threshold", torch.tensor(stop_threshold, dtype=torch.float32))

    @property
    def r(self):
        return self.decoder.r.item()

    @r.setter
    def r(self, value):
        self.decoder.r = self.decoder.r.new_tensor(value, requires_grad=False)

    def forward(self, x, m, speaker_embedding):
        device = next(self.parameters()).device

        self.step += 1
        batch_size, _, steps  = m.size()
        
        attn_hidden = torch.zeros(batch_size, self.decoder_dims, device=device)
        rnn1_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        hidden_states = (attn_hidden, rnn1_hidden, rnn2_hidden)

        rnn1_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        cell_states = (rnn1_cell, rnn2_cell)

        go_frame = torch.zeros(batch_size, self.n_mels, device=device)
        context_vec = torch.zeros(batch_size, self.encoder_dims + self.speaker_embedding_size, device=device)

        encoder_seq = self.encoder(x, speaker_embedding)
        encoder_seq_proj = self.encoder_proj(encoder_seq)
        
        mel_outputs, attn_scores, stop_outputs = [], [], []

        for t in range(0, steps, self.r):
            prenet_in = m[:, :, t - 1] if t > 0 else go_frame
            mel_frames, scores, hidden_states, cell_states, context_vec, stop_tokens = \
                self.decoder(encoder_seq, encoder_seq_proj, prenet_in,
                             hidden_states, cell_states, context_vec, t, x)
            mel_outputs.append(mel_frames)
            attn_scores.append(scores)
            stop_outputs.extend([stop_tokens] * self.r)

        mel_outputs = torch.cat(mel_outputs, dim=2)

        postnet_out = self.postnet(mel_outputs)
        linear = self.post_proj(postnet_out)
        linear = linear.transpose(1, 2)

        attn_scores = torch.cat(attn_scores, 1)
        stop_outputs = torch.cat(stop_outputs, 1)

        return mel_outputs, linear, attn_scores, stop_outputs

    def generate(self, x, speaker_embedding=None, steps=2000):
        self.eval()
        device = next(self.parameters()).device
        
        batch_size, _  = x.size()
        
        attn_hidden = torch.zeros(batch_size, self.decoder_dims, device=device)
        rnn1_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        hidden_states = (attn_hidden, rnn1_hidden, rnn2_hidden)

        rnn1_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        cell_states = (rnn1_cell, rnn2_cell)

        go_frame = torch.zeros(batch_size, self.n_mels, device=device)
        context_vec = torch.zeros(batch_size, self.encoder_dims + self.speaker_embedding_size, device=device)

        encoder_seq = self.encoder(x, speaker_embedding)
        encoder_seq_proj = self.encoder_proj(encoder_seq)
        mel_outputs, attn_scores, stop_outputs = [], [], []

        for t in range(0, steps, self.r):
            
            prenet_in = mel_outputs[-1][:, :, -1] if t > 0 else go_frame
            mel_frames, scores, hidden_states, cell_states, context_vec, stop_tokens = \
            self.decoder(encoder_seq, encoder_seq_proj, prenet_in,
                         hidden_states, cell_states, context_vec, t, x)
            
            mel_outputs.append(mel_frames)
            attn_scores.append(scores)
            stop_outputs.extend([stop_tokens] * self.r)
            
            # Stop the loop when all stop tokens in batch exceed threshold
            if (stop_tokens > 0.5).all() and t > 10: 
                break
            
        mel_outputs = torch.cat(mel_outputs, dim=2)
        postnet_out = self.postnet(mel_outputs)
        linear = self.post_proj(postnet_out)

        linear = linear.transpose(1, 2)
        attn_scores = torch.cat(attn_scores, 1)
        stop_outputs = torch.cat(stop_outputs, 1)

        self.train()

        return mel_outputs, linear, attn_scores

    def init_model(self):
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)

    def get_step(self):
        return self.step.data.item()

    def reset_step(self):
        self.step = self.step.data.new_tensor(1)

    def num_params(self, print_out=True):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        
        if print_out:
            print("Trainable Parameters: %.3fM" % parameters)
        
        return parameters