from core.utils.configs.audio import AudioConfig


class Text2SpeechAudioConfig(AudioConfig):
    N_MELS = 80
    SAMPLE_RATE = 16000
    N_FFT = 2048
    FRAME_SHIFT = 0.0125
    FRAME_LENGTH = 0.05
    REF_LEVEL_DB = 20
    HOP_LENGTH = int(SAMPLE_RATE * FRAME_SHIFT)
    WIN_LENGTH = int(SAMPLE_RATE * FRAME_LENGTH)
    PRE_EMPHASIS = 0.97
    POWER = 1.2
    FMIN = 90
    FMAX = 7600
    ZERO_THRESHOLD = 1e-5
    MIN_LEVEL_DB = -100
    N_ITER = 60
    MAX_DB = 100

