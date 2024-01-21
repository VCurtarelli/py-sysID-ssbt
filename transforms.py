import numpy as np
from scipy.fft import fft, ifft
import scipy.signal as signal


def stft(x, window='hann', nperseg=32, **kwargs):
    _, _, X_stft = signal.stft(x, window=window, return_onesided=False, nperseg=nperseg, **kwargs)
    return X_stft


def istft(X, window='hann', nperseg=32, **kwargs):
    t, xt = signal.istft(X, input_onesided=False, window=window, nperseg=nperseg, **kwargs)
    return xt


def ssbt(x, k0=0.25, window='hann', nperseg=32, **kwargs):
    """
        Single Sideband Transform.
    """
    _, _, X_stft = stft(x, window=window, return_onesided=False, nperseg=nperseg, **kwargs)
    X = np.sqrt(2) * np.real(np.exp(1j*3*np.pi*k0) * X_stft)
    return X


def issbt(X_ssbt, k0=0.25, window='hann', nperseg=32, **kwargs):
    """
        Inverse Single Sideband Transform.
    """
    X_stft = np.sqrt(2) * np.exp(-1j*3*np.pi*k0) * X_ssbt
    t, xt = istft(X_stft, window=window, input_onesided=False, nperseg=nperseg, **kwargs)
    xt = np.real(xt)
    return xt


def rft(xt, k0=0.25):
    """
        Real Fourier Transform.
        Similar to the SSBT, but without considering windowing. Applied to the whole signal.
    """
    X_fft = fft(xt)
    X_rft = np.sqrt(2) * np.real(np.exp(1j*3*np.pi*k0) * X_fft)
    return X_rft


def irft(X_rft, k0=0.25):
    """
        Inverse Real Fourier Transform.
        Similar to the ISSBT, but without considering windowing. Applied to the whole signal.
    """
    X_fft = np.sqrt(2) * np.exp(-1j*3*np.pi*k0) * X_rft
    xt = np.real(ifft(X_fft))
    return xt

