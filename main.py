import os

import numpy as np
import scipy.linalg
from scipy.linalg import inv, det, svd
from functions import *
import scipy.special as spsp
import scipy as sp
from scipy.io import wavfile, savemat, loadmat
import scipy.signal as signal
from scipy.fft import fft, ifft
from transforms import stft, istft, ssbt, issbt, rft, irft
import gen_palette as gp
import soundfile as sf
import matplotlib.pyplot as plt
import sys
from multiprocessing import Pool
import pickle
import time
# from numba import jit, njit
import cupy as cp


np.set_printoptions(suppress=True, precision=12, threshold=sys.maxsize)
timers = {}


def get_time(label):
    timers[label] = time.time()
    if len(timers) > 1:
        show_time(slice(-1,  None))
    
    
def show_time(log=None):
    if log is None:
        log = slice(None, None)
        print()
    keys = list(timers.keys())
    d_times = [timers[keys[i+1]] - timers[keys[i]] for i in range(len(keys)-1)]
    text = ['{}: {:2.4f} sec'.format(keys[i], d_times[i]) for i in range(len(d_times))]
    text = text[log]
    print('\n'.join(text))
    

def sim_parser(comb):
    gen_data(freq_mode=comb[0], signal_mode= 'load', nperseg =comb[1])
    show_time()
    

def gen_data(freq_mode: str = 'stft', signal_mode='random', nperseg=32):
    print('Stt:', freq_mode.upper(), nperseg)
    get_time('Load')
    """
    Parameters
    ----------
    freq_mode: str
        Which to use, STFT or SSBT. Defaults to STFT.
    sig_mode: str
        Which signals to use, 'random' or 'load'. Defaults to 'random'.
    n_per_seg: int
        Number of samples per window in transforms. Defaults to 32.
    Returns
    -------

    """
    
    # Info: Abbreviations
    #       RIR         : Room Impulse Response
    #       RFR         : Room Frequency Response
    #       SIR         : Signal-to-Interference Ratio
    #       SNR         : Signal-to-Noise Ratio
    #       TF          : Transfer Function
    #       RFR         : Relative Transfer Function
    #       FT          : Frequency Transform
    #       FFT         : Fast Fourier Transform
    #       STFT        : Short-Time Fourier Transform
    #       RFT         : Real Fourier Transform
    #       SSBT        : Single-SideBand Transform
    #       GEFT        : GEneric Frequency Transform
    
    """
        ------------------
        - Pre-processing -
        ------------------
    """
    freq_mode = freq_mode.lower()
    if freq_mode not in ['nssbt', 'stft', 'tssbt']:
        raise SyntaxError('Invalid frequency mode.')
    
    """
        -------------
        - Constants -
        -------------
    """
    # Vars: Constants
    #       dx          : Distance between sensors [m]
    #       c           : Wave speed [m/s]
    #       fs          : Sampling frequency [samples/s]
    #       n_sensors   : Number of sensors in array [scalar]
    #       len_rir     : Length of RIRs [samples]
    #       n_bins      : Number of frequency bins after freq. transform [scalar]
    #       n_bins_star : Number of frequency bins after STFT [scalar]
    #       geft        : GEFT - GEneric Frequency Transform {function}
    #       n_win_rir   : Number of windows in the freq. transform of the RIR [scalar]
    #       sym_freqs   : Simulated frequencies {vector} [Hz]
    #       win_p_fil   : Number of windows to use to calculate each filter [scalar]
    #       dist_fil    : How long to wait to calculate filter again [scalar]
    #       array       : Array of sensors {class Array}
    #       epsilon     : Small value to mitigate divide-by-zero errors [scalar]
    #       SIR_in      : Input SIR [scalar, dB]
    #       SNR_in      : Input SNR [scalar, dB]
    
    variables = loadmat('io_input/variables.mat')
    
    dx = variables['delta_x'].item()
    c = variables['c'].item()
    # fs = variables['fs'].item()
    fs = 8000
    len_rir = variables['n'].item()
    
    global n_bins, F_lk_star, geft
    match freq_mode:
        case 'nssbt' | 'tssbt':
            n_bins = nperseg
            geft = ssbt
        
        case 'stft':
            n_bins = (1 + nperseg) // 2 + 1
            geft = stft
        
    noverlap = nperseg//2
    win_type = 'hann'
    window = signal.get_window(win_type, nperseg)
    
    """
        -------------------------
        - Time-domain variables -
        -------------------------
    """
    # Vars: Sources and signals (time)
    #       h_n         : Speech signal's RIR, for each sensor {matrix} [scalar]
    #       g_n         : Interfering signal's RIR, for each sensor {matrix} [scalar]
    #       x_n         : Speech signal at source {vector} [scalar]
    #       v_n         : Interfering signal at source {vector} [scalar]
    #       r_n         : Noise signal, for each sensor {vector} [scalar]
    #       len_x       : Number of samples of x_n [scalar]
    get_time('Time-domain')
    
    h_n = np.loadtxt('io_input/rir_dx_.csv', delimiter=',')
    h_n = h_n[0, :].reshape(-1, 1)
    g_n = np.loadtxt('io_input/rir_v2_.csv', delimiter=',')
    
    global x_n, v_n, r_n
    match signal_mode:
        case 'load':
            x_n, samplerate = sf.read('io_input/audio_speech_male_r.flac')
            x_n = resample(x_n, samplerate, fs)
            x_n = x_n[:x_n.size//2]
            babble = scipy.io.loadmat('io_input/babble.mat')
            v_n, samplerate = babble['babble'].reshape(-1), babble['fs']
            v_n = resample(v_n, samplerate, fs)
            r_n, samplerate = sf.read('io_input/audio_noise_wgn.flac')
            r_n = resample(r_n, samplerate, fs).reshape(-1, 1)
            h_n = resample(h_n, 16000, fs).reshape(-1, 1)
        
        case 'random' | _:
            len_x = 100000
            x_n = np.random.rand(len_x)
            v_n = np.random.rand(2 * len_x)
            r_n = np.random.rand(2 * len_x)
    
    len_rir = h_n.size
    n_win_H = int(np.ceil((len_rir + nperseg - 1) / noverlap))
    """
        -------------------------------
        - Time-Freq. domain variables -
        -------------------------------
    """
    # Vars: Input sources and signals (freq.)
    #       H_lkk       : Speech signal's crossband TF, for each sensor {tensor} [scalar]
    get_time('Freq-domain')
    
    H_lkk = np.empty((n_win_H, nperseg, nperseg), dtype=complex)
    h_k = np.empty((n_win_H*nperseg, nperseg), dtype=complex)
    for k1 in range(nperseg):
        for k2 in range(nperseg):
            phi_kk_n = np.empty((nperseg,), dtype=complex)
            for n in range(nperseg):
                s = window[:(nperseg-n)] * window[n:] * np.exp(-1j*2*np.pi/nperseg * (k1-k2) * np.arange(nperseg-n))
                s = np.sum(s)
                phi_kk_n[n] = np.exp(1j*2*np.pi/nperseg * k2 * n) * s
            H_lkk[:, k1, k2] = np.convolve(h_n.reshape(-1), phi_kk_n.reshape(-1))[::noverlap]
            h_k[n_win_H*k2:n_win_H*(k2+1), k1] = H_lkk[:, k1, k2]
    
    X_lk = tr(stft(x_n, window=win_type, nperseg=nperseg))
    n_win_X = X_lk.shape[0]
    n_win_Y = n_win_X
    
    X = np.empty((n_win_Y, nperseg*n_win_H), dtype=complex)
    for l in range(n_win_Y):
        x_l = np.empty((nperseg*n_win_H,), dtype=complex)
        for k_ in range(nperseg):
            xd_lk = np.zeros((n_win_H,), dtype=complex)
            for l_ in range(n_win_H):
                xd_lk[l_] = X_lk[l-l_, k_]
                if l-l_ <= 0:
                    break
            x_l[k_*n_win_H:(k_+1)*n_win_H] = xd_lk
        X[l, :] = x_l
        
    K = 5
    hp_k = np.zeros(((2*K+1)*n_win_H, nperseg), dtype=complex)
            
    V_lk = tr(stft(v_n, window = win_type, nperseg = nperseg))
    V_lk = V_lk[:n_win_Y, :]
    
    Y_lk = np.zeros((n_win_Y, nperseg), dtype = complex)
    
    for k in range(nperseg):
        Y_lk[:, k] = X @ h_k[:, k] + V_lk[:, k]
    
    # Info:
    #   Finding the solution to hp_k_opt \equiv \min_{h_k} ||y_k - Xp_k hp_k||^2
    #   Using Singular-Value Decomposition:
    #       Xp_k = U @ np.diag(S) @ Vh
    #       U and Vh are singular (inverse = hermitian), S is diagonal of single values
    #       Therefore, if rank(Xp_k) = min(Xp_k.shape), this produces its inverse
    #       If rank(Xp_k) < min(Xp_k.shape), this produces the minimum-norm solution to the problem
    #   See:
    #       https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
    #       https://www.physicsforums.com/threads/linear-least-square-method-for-singular-matrices.446868/
    get_time('Calc solution')
    data = {'X_lk': X_lk,
            'V_lk': V_lk,
            'Y_lk': Y_lk,
            'X': X,
            'h_k': h_k,
            'nperseg': nperseg}
    
    X_ = np.hstack([X, X, X])
    for k in range(nperseg):
        Xp_k = np.array(X_[:, (k-K+nperseg)*n_win_H:(k+K+nperseg+1)*n_win_H])
        U, S, Vh = np.linalg.svd(Xp_k, full_matrices=False)
        n = np.linalg.matrix_rank(Xp_k)
        U_ = U[:, :n]
        S_ = S[:n]
        Vh_ = Vh[:n, :]
        A_inv = (he(Vh_) / S_) @ he(U_)
        hp_k[:, k] = np.matmul(A_inv, Y_lk[:, k])
    data['hp_k'] = hp_k
    get_time('End')
    
    with open('io_output/data.pckl', 'wb') as file:
        pickle.dump(data, file)


def main():
    freqmodes = [
        'stft',
        # 'nssbt',
        # 'tssbt'
    ]

    npersegs = [
        32,
        # 64,
    ]

    combs = [(freqmode, nperseg) for freqmode in freqmodes for nperseg in npersegs]
    ncombs = min(len(combs), 4)
    parallel = False
    
    if parallel:
        with Pool(ncombs) as p:
            p.map(sim_parser, combs)
    else:
        for comb in combs:
            sim_parser(comb)


if __name__ == '__main__':
    main()
