from core.utils.configs.audio import AudioConfig


class SpeakerEncoderAudioConfig(AudioConfig):
    N_MELS = 80
    SAMPLE_RATE = 16000
    FRAME_SHIFT = 0.01
    FRAME_LENGTH = 0.025
    HOP_LENGTH = int(SAMPLE_RATE * FRAME_SHIFT)
    WIN_LENGTH = int(SAMPLE_RATE * FRAME_LENGTH)
    N_FFT = 1024
    FMIN = 90
    FMAX = 7600
    ZERO_THRESHOLD = 1e-5
    MIN_AMPLITUDE = 0.3
    MAX_AMPLITUDE = 1.0
    MIN_LEVEL_DB = -100
    REF_LEVEL_DB = 0
    NUM_FRAMES = 128
    SCALING_FACTOR = 0.95
