from dataclasses import dataclass

@dataclass
class VoiceEnhancementConfig:
    # Audio parameters
    SAMPLE_RATE: int = 8000
    MIN_DURATION: float = 1.0
    FORMAT: str = "WAV"
    
    # Spectrogram parameters
    FRAME_LENGTH: int = 8064
    HOP_LENGTH_FRAME: int = 8064
    N_FFT: int = 255
    HOP_LENGTH_FFT: int = 63
    DIM_SQUARE_SPEC: int = int(N_FFT / 2) + 1

    # Visualization parameters
    SPEC_FIGSIZE: tuple = (10, 5)
    WAVE_FIGSIZE: tuple = (10, 4)
    DPI: int = 100