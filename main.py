import os

import matplotlib.colors
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

np.set_printoptions(suppress = True, precision = 12, threshold = sys.maxsize)
timers = {}


def get_time(label):
    timers[label] = time.time()
    if len(timers) > 1:
        show_time(slice(-1, None))


def reset_time():
    timers.clear()


def show_time(log=None):
    if log is None:
        log = slice(None, None)
        print()
    keys = list(timers.keys())
    d_times = [timers[keys[i + 1]] - timers[keys[i]] for i in range(len(keys) - 1)]
    text = ['{}: {:2.4f} sec'.format(keys[i], d_times[i]) for i in range(len(d_times))]
    text = text[log]
    print('\n'.join(text))


def sim_parser(comb):
    gen_data(comb)
    # show_time()
    print()


def gen_data(comb, signal_mode='load', fs=8000):
    freq_mode, SNR, K, nperseg = comb
    print('Stt: Mode = {} | SNR = {:+}dB | K = {} | N = {}'.format(freq_mode.upper(), SNR, K, nperseg))
    reset_time()
    get_time('Load')
    """
    Parameters
    ----------
    freq_mode: str
        Frequency transform to use. Defaults to 'stft'.
    SNR: int
        Signal-to-Noise Ratio *at sensor*.
    K: str
        Number of cross-band filters to use.
    nperseg: int
        Number of samples per window in transforms. Defaults to 32.
    signal_mode: int
        How to do signals. To load or to generate random.
    fs: int
        Target sampling frequency.
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
    
    if freq_mode not in ['ssbt', 'stft']:
        raise AssertionError('Unexpected transform.')
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
    len_rir = variables['n'].item()
    
    global F_lk_star, geft, igeft
    match freq_mode:
        case 'ssbt':
            geft = ssbt
            igeft = issbt
        
        case 'stft':
            geft = stft
            igeft = istft
    
    noverlap = nperseg // 2
    win_type = 'hamming'
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
    
    h_n = np.loadtxt('io_input/rir_dx_.csv', delimiter = ',')
    h_n = h_n[0, :].reshape(-1, 1)
    g_n = np.loadtxt('io_input/rir_v2_.csv', delimiter = ',')
    g_n = g_n[0, :].reshape(-1, 1)
    
    global x_n, v_n, v_n
    match signal_mode:
        case 'load':
            len_x = 30000
            x_n, samplerate = sf.read('io_input/audio_noise_wgn.flac')
            x_n = resample(x_n, samplerate, fs)
            x_n = x_n[2 * len_x:3 * len_x]
            # babble = scipy.io.loadmat('io_input/babble.mat')
            # v_n, samplerate = babble['babble'].reshape(-1), babble['fs']
            # v_n = resample(v_n, samplerate, fs)
            v_n, samplerate = sf.read('io_input/audio_noise_wgn.flac')
            v_n = resample(v_n, samplerate, fs).reshape(-1, 1)
            h_n = resample(h_n, 16000, fs).reshape(-1, 1)
        
        case 'random' | _:
            len_x = 30000
            x_n = np.random.randn(len_x)
            v_n = np.random.randn(2 * len_x)
            v_n = np.random.randn(2 * len_x)
    
    len_rir = h_n.size
    n_win_H = int(np.ceil((len_rir + nperseg - 1) / noverlap) + np.ceil(nperseg / noverlap) - 1)
    h_coefs = np.arange(n_win_H)
    """
        -------------------------------
        - Time-Freq. domain variables -
        -------------------------------
    """
    # Vars: Input sources and signals (freq.)
    #       H_lkk       : Speech signal's crossband TF, for each sensor {tensor} [scalar]
    get_time('Freq-domain')
    
    H_lkk = np.zeros((n_win_H, nperseg, nperseg), dtype = complex)
    h_k = np.zeros((n_win_H * nperseg, nperseg), dtype = complex)
    for k1 in range(nperseg):
        for k2 in range(nperseg):
            phi_kk_n = np.zeros((2 * nperseg - 1,))
            match freq_mode:
                case 'stft':
                    s1 = window * np.exp(-1j * 2 * np.pi / nperseg * np.arange(nperseg) * (k1 - k2))
                    s2 = window
                    phi_kk_n = np.correlate(s1, s2, mode = 'full') * np.exp(
                        1j * 2 * np.pi / nperseg * k1 * np.arange(-nperseg + 1, nperseg))
                case 'ssbt':
                    a1 = 2 * np.pi * k2 / nperseg * np.arange(nperseg)
                    s1 = window * 1 / np.sqrt(2) * (np.cos(a1) + np.sin(a1))
                    a2 = 2 * np.pi * k1 / nperseg * np.arange(nperseg)
                    s2 = window * 1 / np.sqrt(2) * (np.cos(a2) + np.sin(a2))
                    phi_kk_n = np.correlate(s1, s2, mode = 'full')
            
            S = np.convolve(h_n.reshape(-1), phi_kk_n.reshape(-1))[::noverlap]
            S = S[:n_win_H]
            H_lkk[:, k1, k2] = S
            h_k[n_win_H * k2:n_win_H * (k2 + 1), k1] = H_lkk[:, k1, k2]
    
    X_lk = tr(geft(x_n, window = win_type, nperseg = nperseg))
    n_win_X = X_lk.shape[0]
    n_win_Y = n_win_X
    
    X = np.zeros((n_win_Y, nperseg * n_win_H), dtype = complex)
    for l in range(n_win_H):
        for k in range(nperseg):
            X[l:, k * n_win_H + l] = X_lk[:(n_win_Y - l), k]
    
    V_lk = tr(geft(v_n.reshape(-1), window = win_type, nperseg = nperseg))
    V_lk = V_lk[:n_win_Y, :]
    
    Y_lk = np.zeros((n_win_Y, nperseg), dtype = complex)
    S_lk = np.zeros((n_win_Y, nperseg), dtype = complex)
    
    for k in range(nperseg):
        S = X @ h_k[:, k]
        V = V_lk[:, k]
        std_S = np.var(S)
        std_V = np.var(V)
        S = (S / std_S) if std_S > 0 else S
        V = (V / std_V / 10 ** (SNR / 20)) if std_V > 0 else V
        W = V
        S_lk[:, k] = S
        Y_lk[:, k] = S + W
    get_time('Calc solution')
    
    hp_k = np.zeros(((2 * K + 1) * n_win_H, nperseg), dtype = complex)
    X_ = np.hstack([X, X, X])
    for k in range(nperseg):
        Xp_k = np.array(X_[:, (k - K + nperseg) * n_win_H:(k + K + nperseg + 1) * n_win_H])
        mat = he(Xp_k) @ Xp_k
        hp_k[:, k] = inv(mat) @ he(Xp_k) @ Y_lk[:, k]
    get_time('Save data')
    
    d_k = np.zeros((n_win_Y, nperseg), dtype = complex)
    dp_k = np.zeros((n_win_Y, nperseg), dtype = complex)
    for k in range(nperseg):
        d_k[:, k] = X @ h_k[:, k]
        Xp_k = np.array(X_[:, (k - K + nperseg) * n_win_H:(k + K + nperseg + 1) * n_win_H])
        dp_k[:, k] = Xp_k @ hp_k[:, k]
    
    d_n = igeft(tr(d_k), nperseg = nperseg, window = win_type)
    dp_n = igeft(tr(dp_k), nperseg = nperseg, window = win_type)
    get_time('End')
    data = {'X_lk': X_lk,
            'V_lk': V_lk,
            'Y_lk': Y_lk,
            'h_k': h_k,
            'nperseg': nperseg,
            'hp_k': hp_k,
            'd_k': d_k,
            'dp_k': dp_k,
            'd_n': d_n,
            'dp_n': dp_n,
            'time': timers[list(timers.keys())[-1]] - timers[list(timers.keys())[0]]
            }
    
    name = ['data', freq_mode, '{:.2f}'.format(SNR), '{}'.format(K), '{}'.format(nperseg)]
    filename = 'io_output/var_data/{}'.format('__'.join(name))
    with open(filename + '.pckl', 'wb') as file:
        pickle.dump(data, file)
    savemat(filename + '.mat', data)
    

def proc_data_m(freq_modes, SNRs, Ks, nperseg):
    ERLEs = {}
    linestyles = [(0, (4, 2*i)) for i, _ in enumerate(freq_modes)]
    hsv_colors = [(i/len(Ks), 0.6, 0.8) for i, _ in enumerate(Ks)]
    linewidths = [1, 2]
    alphas = [1, 0.5]
    for K_idx, K in enumerate(Ks):
        for fm_idx, freq_mode in enumerate(freq_modes):
            ERLE_K = []
            for SNR in SNRs:
                name = ['data', freq_mode, '{:.2f}'.format(SNR), '{}'.format(K), '{}'.format(nperseg)]
                filename = 'io_output/var_data/{}'.format('__'.join(name))
                with open(filename + '.pckl', 'rb') as file:
                    data = pickle.load(file)
                d_n = data['d_n']
                dp_n = data['dp_n']
                dp_n = np.nan_to_num(dp_n)
                
                d_n_norm = d_n / np.std(d_n)
                dp_n_norm = dp_n / np.std(dp_n)
                
                ERLE = np.var(d_n_norm) / np.var(d_n_norm - dp_n_norm)
                ERLE = dB(ERLE)
                # print(ERLE)
                ERLE_K.append(ERLE)
            key = 'K={}, {}'.format(K, freq_mode.upper())
            ERLEs[key] = ERLE_K
            
            linestyle = linestyles[fm_idx]
            color = matplotlib.colors.hsv_to_rgb(hsv_colors[K_idx])
            linewidth = linewidths[fm_idx]
            alpha = alphas[fm_idx]
            plt.plot(SNRs, ERLE_K, label = key, linestyle = linestyle, color = color, linewidth=linewidth, alpha=alpha)
            
            data = ['snr, val']
            for idx, _ in enumerate(SNRs):
                data.append('{}, {:.6f}'.format(SNRs[idx], ERLE_K[idx]))
            data = '\n'.join(data)
            with open('io_output/plots/ERLE__{}__K_{}.csv'.format(freq_mode, K), 'w') as f:
                f.write(data)
                f.close()
    
    with open('io_output/plots/plots.pckl', 'wb') as file:
        pickle.dump(ERLEs, file)
    
    plt.legend(loc = 'upper left')
    # plt.show()
    
    pass


def main():
    SNRs = range(-40, 20 + 1, 5)
    Ks = range(0, 4 + 1)
    freq_modes = (
        'stft',
        'ssbt',
    )
    npersegs = (
        # 32,
        64,
    )
    
    combs = [(freq_mode, SNR, K, nperseg) for freq_mode in freq_modes for SNR in SNRs for K in Ks for nperseg in npersegs]
    ncombs = min(len(combs), 4)
    
    data_modes = {
        1: 'gen',
        3: 'proc_m'
    }
    
    idx = 3
    data_mode = data_modes[idx]
    match data_mode:
        case 'gen':
            parallel = False
            if parallel:
                with Pool(ncombs) as p:
                    p.map(sim_parser, combs)
            else:
                for comb in combs:
                    sim_parser(comb)
        case 'proc_m':
            for nperseg in npersegs:
                proc_data_m(freq_modes, SNRs, Ks, nperseg)
            plt.show()


if __name__ == '__main__':
    main()
