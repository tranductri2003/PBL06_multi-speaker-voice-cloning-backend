import librosa
import librosa.filters
import numpy as np
from scipy import signal
from scipy.io import wavfile
import soundfile as sf

import torch
from pathlib import Path
from typing import Union, List
import ast
import pprint

from text_to_speech.models.tacotron import Tacotron
from core.utils.text2sequence.en import EnglishText2Sequence
from core.settings import EN_TACOTRON_PARAMS

class HParams(object):
    def __init__(self, **kwargs): self.__dict__.update(kwargs)
    def __setitem__(self, key, value): setattr(self, key, value)
    def __getitem__(self, key): return getattr(self, key)
    def __repr__(self): return pprint.pformat(self.__dict__)

    def parse(self, string):
        # Overrides hparams from a comma-separated string of name=value pairs
        if len(string) > 0:
            overrides = [s.split("=") for s in string.split(",")]
            keys, values = zip(*overrides)
            keys = list(map(str.strip, keys))
            values = list(map(str.strip, values))
            for k in keys:
                self.__dict__[k] = ast.literal_eval(values[keys.index(k)])
        return self

hparams = HParams(
        ### Signal Processing (used in both synthesizer and vocoder)
        sample_rate = 16000,
        n_fft = 800,
        num_mels = 80,
        hop_size = 200,                             # Tacotron uses 12.5 ms frame shift (set to sample_rate * 0.0125)
        win_size = 800,                             # Tacotron uses 50 ms frame length (set to sample_rate * 0.050)
        fmin = 55,
        min_level_db = -100,
        ref_level_db = 20,
        max_abs_value = 4.,                         # Gradient explodes if too big, premature convergence if too small.
        preemphasis = 0.97,                         # Filter coefficient to use if preemphasize is True
        preemphasize = True,

        ### Tacotron Text-to-Speech (TTS)
        tts_embed_dims = 512,                       # Embedding dimension for the graphemes/phoneme inputs
        tts_encoder_dims = 256,
        tts_decoder_dims = 128,
        tts_postnet_dims = 512,
        tts_encoder_K = 5,
        tts_lstm_dims = 1024,
        tts_postnet_K = 5,
        tts_num_highways = 4,
        tts_dropout = 0.5,
        tts_stop_threshold = -3.4,                  # Value below which audio generation ends.
                                                    # For example, for a range of [-4, 4], this
                                                    # will terminate the sequence at the first
                                                    # frame that has all values < -3.4

        ### Tacotron Training
        tts_schedule = [(2,  1e-4,  20_000,  12),   # Progressive training schedule
                        (2,  5e-5,  40_000,  12),   # (r, lr, step, batch_size)
                        (2,  2e-5,  80_000,  12),   #
                        (2,  1e-5, 160_000,  12),   # r = reduction factor (# of mel frames
                        (2,  1e-5, 320_000,  12),   #     synthesized for each decoder iteration)
                        (2,  1e-5, 640_000,  12)],  # lr = learning rate

        tts_clip_grad_norm = 1.0,                   # clips the gradient norm to prevent explosion - set to None if not needed
        tts_eval_interval = 500,                    # Number of steps between model evaluation (sample generation)
                                                    # Set to -1 to generate after completing epoch, or 0 to disable

        tts_eval_num_samples = 1,                   # Makes this number of samples

        ### Data Preprocessing
        max_mel_frames = 900,
        rescale = True,
        rescaling_max = 0.9,
        synthesis_batch_size = 16,                  # For vocoder preprocessing and inference.

        ### Mel Visualization and Griffin-Lim
        signal_normalization = True,
        power = 1.5,
        griffin_lim_iters = 60,

        ### Audio processing options
        fmax = 7600,                                # Should not exceed (sample_rate // 2)
        allow_clipping_in_normalization = True,     # Used when signal_normalization = True
        clip_mels_length = True,                    # If true, discards samples exceeding max_mel_frames
        use_lws = False,                            # "Fast spectrogram phase recovery using local weighted sums"
        symmetric_mels = True,                      # Sets mel range to [-max_abs_value, max_abs_value] if True,
                                                    #               and [0, max_abs_value] if False
        trim_silence = True,                        # Use with sample_rate of 16000 for best results

        ### SV2TTS
        speaker_embedding_size = 256,               # Dimension for the speaker embedding
        silence_min_duration_split = 0.4,           # Duration in seconds of a silence for an utterance to be split
        utterance_min_duration = 1.6,               # Duration in seconds below which utterances are discarded
        )


class Synthesizer:
    sample_rate = hparams.sample_rate
    hparams = hparams

    def __init__(self, model_fpath=None, model=None, verbose=True):
        self.model_fpath = model_fpath
        self.verbose = verbose

        self.device = torch.device("cpu")
            
        if self.verbose:
            print("Synthesizer using device:", self.device)

        self._model = model

    def is_loaded(self):
        return self._model is not None

    def load(self):

        self._model = Tacotron(**EN_TACOTRON_PARAMS).to(self.device)

        self._model.load(self.model_fpath)
        self._model.eval()

    def synthesize_spectrograms(self, texts: List[str],
                                embeddings: Union[np.ndarray, List[np.ndarray]],
                                return_alignments=False):
        if not self.is_loaded():
            self.load()

        # Preprocess text inputs
        t2seq = EnglishText2Sequence()
        print(texts)
        inputs = [t2seq(text.strip()) for text in texts]
        if not isinstance(embeddings, list):
            embeddings = [embeddings]

        # Batch inputs
        batched_inputs = [inputs[i:i+hparams.synthesis_batch_size]
                             for i in range(0, len(inputs), hparams.synthesis_batch_size)]
        batched_embeds = [embeddings[i:i+hparams.synthesis_batch_size]
                             for i in range(0, len(embeddings), hparams.synthesis_batch_size)]

        specs = []
        for i, batch in enumerate(batched_inputs, 1):
            if self.verbose:
                print(f"\n| Generating {i}/{len(batched_inputs)}")

            # Pad texts so they are all the same length
            text_lens = [len(text) for text in batch]
            max_text_len = max(text_lens)
            chars = [pad1d(text, max_text_len) for text in batch]
            chars = np.stack(chars)

            # Stack speaker embeddings into 2D array for batch processing
            speaker_embeds = np.stack(batched_embeds[i-1])

            # Convert to tensor
            chars = torch.tensor(chars).long().to(self.device)
            speaker_embeddings = torch.tensor(speaker_embeds).float().to(self.device)

            # Inference
            _, mels, alignments = self._model.generate(chars, speaker_embeddings)
            mels = mels.detach().cpu().numpy()
            for m in mels:
                while np.max(m[:, -1]) < hparams.tts_stop_threshold:
                    m = m[:, :-1]
                specs.append(m)

        if self.verbose:
            print("\n\nDone.\n")
        return (specs, alignments) if return_alignments else specs

    @staticmethod
    def load_preprocess_wav(fpath):
        wav = librosa.load(str(fpath), hparams.sample_rate)[0]
        if hparams.rescale:
            wav = wav / np.abs(wav).max() * hparams.rescaling_max
        return wav

    @staticmethod
    def make_spectrogram(fpath_or_wav: Union[str, Path, np.ndarray]):
        if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
            wav = Synthesizer.load_preprocess_wav(fpath_or_wav)
        else:
            wav = fpath_or_wav

        mel_spectrogram = melspectrogram(wav, hparams).astype(np.float32)
        return mel_spectrogram

    @staticmethod
    def mel_to_audio_using_griffin_lim(mel):
        return inv_mel_spectrogram(mel, hparams)
    
    @staticmethod
    def mag_to_audio_using_griffin_lim(mag):
        return inv_linear_spectrogram(mag, hparams)

def pad1d(x, max_len, pad_value=0):
    return np.pad(x, (0, max_len - len(x)), mode="constant", constant_values=pad_value)


def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]

def save_wav(wav, path, sr):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    wavfile.write(path, sr, wav.astype(np.int16))

def save_wavenet_wav(wav, path, sr):
    sf.write(path, wav.astype(np.float32), sr)

def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav

def inv_preemphasis(wav, k, inv_preemphasize=True):
    if inv_preemphasize:
        return signal.lfilter([1], [1, -k], wav)
    return wav

def start_and_end_indices(quantized, silence_threshold=2):
    for start in range(quantized.size):
        if abs(quantized[start] - 127) > silence_threshold:
            break
    for end in range(quantized.size - 1, 1, -1):
        if abs(quantized[end] - 127) > silence_threshold:
            break
    
    assert abs(quantized[start] - 127) > silence_threshold
    assert abs(quantized[end] - 127) > silence_threshold
    
    return start, end

def get_hop_size(hparams):
    hop_size = hparams.hop_size
    if hop_size is None:
        assert hparams.frame_shift_ms is not None
        hop_size = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
    return hop_size

def linearspectrogram(wav, hparams):
    D = _stft(preemphasis(wav, hparams.preemphasis, hparams.preemphasize), hparams)
    S = _amp_to_db(np.abs(D), hparams) - hparams.ref_level_db
    
    if hparams.signal_normalization:
        return _normalize(S, hparams)
    return S

def melspectrogram(wav, hparams):
    D = _stft(preemphasis(wav, hparams.preemphasis, hparams.preemphasize), hparams)
    S = _amp_to_db(_linear_to_mel(np.abs(D), hparams), hparams) - hparams.ref_level_db
    
    if hparams.signal_normalization:
        return _normalize(S, hparams)
    return S

def inv_linear_spectrogram(linear_spectrogram, hparams):
    if hparams.signal_normalization:
        D = _denormalize(linear_spectrogram, hparams)
    else:
        D = linear_spectrogram
    
    S = _db_to_amp(D + hparams.ref_level_db)
    
    if hparams.use_lws:
        processor = _lws_processor(hparams)
        D = processor.run_lws(S.astype(np.float64).T ** hparams.power)
        y = processor.istft(D).astype(np.float32)
        return inv_preemphasis(y, hparams.preemphasis, hparams.preemphasize)
    else:
        return inv_preemphasis(_griffin_lim(S ** hparams.power, hparams), hparams.preemphasis, hparams.preemphasize)

def inv_mel_spectrogram(mel_spectrogram, hparams):
    if hparams.signal_normalization:
        D = _denormalize(mel_spectrogram, hparams)
    else:
        D = mel_spectrogram
    
    S = _mel_to_linear(_db_to_amp(D + hparams.ref_level_db), hparams)
    
    if hparams.use_lws:
        processor = _lws_processor(hparams)
        D = processor.run_lws(S.astype(np.float64).T ** hparams.power)
        y = processor.istft(D).astype(np.float32)
        return inv_preemphasis(y, hparams.preemphasis, hparams.preemphasize)
    else:
        return inv_preemphasis(_griffin_lim(S ** hparams.power, hparams), hparams.preemphasis, hparams.preemphasize)

def _lws_processor(hparams):
    pass

def _griffin_lim(S, hparams):
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex128)
    y = _istft(S_complex * angles, hparams)
    for i in range(hparams.griffin_lim_iters):
        angles = np.exp(1j * np.angle(_stft(y, hparams)))
        y = _istft(S_complex * angles, hparams)
    return y

def _stft(y, hparams):
    if hparams.use_lws:
        return _lws_processor(hparams).stft(y).T
    else:
        return librosa.stft(y=y, n_fft=hparams.n_fft, hop_length=get_hop_size(hparams), win_length=hparams.win_size)

def _istft(y, hparams):
    return librosa.istft(y, hop_length=get_hop_size(hparams), win_length=hparams.win_size)

def num_frames(length, fsize, fshift):
    pad = (fsize - fshift)
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    return M


def pad_lr(x, fsize, fshift):
    M = num_frames(len(x), fsize, fshift)
    pad = (fsize - fshift)
    T = len(x) + 2 * pad
    r = (M - 1) * fshift + fsize - T
    return pad, pad + r


def librosa_pad_lr(x, fsize, fshift):
    return 0, (x.shape[0] // fshift + 1) * fshift - x.shape[0]


_mel_basis = None
_inv_mel_basis = None

def _linear_to_mel(spectogram, hparams):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis(hparams)
    return np.dot(_mel_basis, spectogram)

def _mel_to_linear(mel_spectrogram, hparams):
    global _inv_mel_basis
    if _inv_mel_basis is None:
        _inv_mel_basis = np.linalg.pinv(_build_mel_basis(hparams))
    return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))

def _build_mel_basis(hparams):
    assert hparams.fmax <= hparams.sample_rate // 2
    return librosa.filters.mel(sr=hparams.sample_rate, n_fft=hparams.n_fft, n_mels=hparams.num_mels,
                               fmin=hparams.fmin, fmax=hparams.fmax)

def _amp_to_db(x, hparams):
    min_level = np.exp(hparams.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))

def _db_to_amp(x):
    return np.power(10.0, (x) * 0.05)

def _normalize(S, hparams):
    if hparams.allow_clipping_in_normalization:
        if hparams.symmetric_mels:
            return np.clip((2 * hparams.max_abs_value) * ((S - hparams.min_level_db) / (-hparams.min_level_db)) - hparams.max_abs_value,
                           -hparams.max_abs_value, hparams.max_abs_value)
        else:
            return np.clip(hparams.max_abs_value * ((S - hparams.min_level_db) / (-hparams.min_level_db)), 0, hparams.max_abs_value)
    
    assert S.max() <= 0 and S.min() - hparams.min_level_db >= 0
    if hparams.symmetric_mels:
        return (2 * hparams.max_abs_value) * ((S - hparams.min_level_db) / (-hparams.min_level_db)) - hparams.max_abs_value
    else:
        return hparams.max_abs_value * ((S - hparams.min_level_db) / (-hparams.min_level_db))

def _denormalize(D, hparams):
    if hparams.allow_clipping_in_normalization:
        if hparams.symmetric_mels:
            return (((np.clip(D, -hparams.max_abs_value,
                              hparams.max_abs_value) + hparams.max_abs_value) * -hparams.min_level_db / (2 * hparams.max_abs_value))
                    + hparams.min_level_db)
        else:
            return ((np.clip(D, 0, hparams.max_abs_value) * -hparams.min_level_db / hparams.max_abs_value) + hparams.min_level_db)
    
    if hparams.symmetric_mels:
        return (((D + hparams.max_abs_value) * -hparams.min_level_db / (2 * hparams.max_abs_value)) + hparams.min_level_db)
    else:
        return ((D * -hparams.min_level_db / hparams.max_abs_value) + hparams.min_level_db)