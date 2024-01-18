import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import stft, istft


def ssbt(x, fs, k0=0.25, **kwargs):
    """
        Single Sideband Transform.
    """
    freqs, win_ts, X_stft = stft(x, fs, return_onesided=False, **kwargs)
    X_ssbt = np.sqrt(2) * np.real(np.exp(1j*3*np.pi*k0) * X_stft)
    return freqs, win_ts, X_ssbt


def issbt(X_ssbt, fs, k0=0.25, **kwargs):
    """
        Inverse Single Sideband Transform.
    """
    X_stft = np.sqrt(2) * np.exp(-1j*3*np.pi*k0) * X_ssbt
    t, xt = istft(X_stft, fs, input_onesided=False, **kwargs)
    xt = np.real(xt)
    return t, xt


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

