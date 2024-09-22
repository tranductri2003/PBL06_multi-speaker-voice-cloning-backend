"""
    File contains all functions for preprocessing audio
"""

import librosa
import numpy as np


def normalize(spectrogram, min_level_db=-100):
    """
    Normalize a spectrogram or audio feature matrix to the range [0, 1].

    This function normalizes the input matrix `S` based on a specified minimum decibel level.
    The normalization is performed by scaling the values such that the minimum level is set to 0 
    and the maximum level is set to 1.

    Args:
        spectrogram (np.ndarray): Input matrix representing a spectrogram,
                        where the values are in decibels (dB).
        min_level_db (float, optional): The minimum level in dB that corresponds to a value of 0 
                                         after normalization. Defaults to -100 dB.

    Returns:
        np.ndarray: A normalized matrix with values clipped to the range [0, 1].
    """
    return np.clip((spectrogram - min_level_db) / -min_level_db, 0, 1)



def linear_to_mel(
    spectrogram, sample_rate=16000, n_fft=1024, fmin=90, fmax=7600, n_mels=80
):
    """
    Convert a linear spectrogram to a Mel spectrogram.

    This function takes a linear spectrogram (magnitude or power) and transforms it into a Mel-scaled spectrogram.
    The Mel scale is designed to approximate the human ear's perception of sound frequencies.

    Args:
        spectrogram (np.ndarray): A linear spectrogram of audio, usually a 2D array where
                                  one dimension represents time and the other represents frequency.
        sample_rate (int, optional): Sample rate of the audio signal. Defaults to 16000 Hz.
        n_fft (int, optional): Number of FFT components. This determines the resolution of the spectrogram.
                               Defaults to 1024.
        fmin (int, optional): Minimum frequency (in Hz) to consider when creating the Mel scale.
                              Defaults to 90 Hz.
        fmax (int, optional): Maximum frequency (in Hz) to consider when creating the Mel scale.
                              Defaults to 7600 Hz.
        n_mels (int, optional): Number of Mel bands to generate. This determines the number of
                                frequency bins in the output. Defaults to 80.

    Returns:
        np.ndarray: A Mel spectrogram corresponding to the input linear spectrogram,
                    with dimensions based on `n_mels` and the time dimension of the input.

    """
    return librosa.feature.melspectrogram(
        S=spectrogram, sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
    )


def amp_to_db(x):
    """
    Convert amplitude values to decibels (dB).

    This function converts amplitude values to the decibel (dB) scale using the formula:
    dB = 20 * log10(amplitude). A small constant (1e-5) is added to avoid log of zero.

    Args:
        x (np.ndarray or scalar): Amplitude values to be converted to decibels.
                                  Can be a scalar or an array.

    Returns:
        np.ndarray or scalar: The corresponding decibel values for the input amplitude values.
    """
    return 20.0 * np.log10(np.maximum(1e-5, x))



def gen_stft(y, n_fft=1024, hop_length=256, win_length=1024):
    """
    Compute the Short-Time Fourier Transform (STFT) of an audio signal.

    This function uses `librosa.stft` to compute the STFT, which transforms a time-domain 
    audio signal into its frequency-domain representation.

    Args:
        y (np.ndarray): Input audio signal (time-domain), typically a 1D array.
        n_fft (int, optional): Number of FFT components. Determines the frequency resolution. 
                               Defaults to 1024.
        hop_length (int, optional): Number of audio samples between successive frames. 
                                    Defaults to 256.
        win_length (int, optional): Each frame of audio is windowed by this number of samples. 
                                    Defaults to 1024.

    Returns:
        np.ndarray: The complex-valued STFT matrix of shape `(n_fft // 2 + 1, t)`, where `t` is 
                    the number of frames, representing the frequency content over time.
    """
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)



def gen_melspectrogram(y, sr=16000, n_fft=1024, hop_length=256, win_length=1024, n_mels=80):
    """
    Generate a Mel-scaled spectrogram from an audio signal.

    This function first computes the STFT of the input signal, converts the linear spectrogram
    to a Mel spectrogram, and then normalizes it by scaling the values between 0 and 1. The 
    amplitude is converted to the decibel scale before normalization.

    Args:
        y (np.ndarray): Input audio signal (time-domain), typically a 1D array.

    Returns:
        np.ndarray: A normalized Mel spectrogram with values clipped between 0 and 1.

    Steps:
        1. Compute the STFT of the input signal.
        2. Convert the STFT's magnitude to a Mel spectrogram.
        3. Convert the amplitude values to decibels.
        4. Normalize the Mel spectrogram and clip values to the range [0, 1].

    """
    stft = gen_stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mel = amp_to_db(linear_to_mel(np.abs(stft), sample_rate=sr, n_fft=n_fft, n_mels=n_mels))
    return np.clip(normalize(mel), 0, 1)
