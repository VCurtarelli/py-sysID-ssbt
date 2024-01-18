import os

import numpy as np
import scipy.linalg
from numpy.linalg import inv, det
from functions import *
import scipy.special as spsp
import scipy as sp
from scipy.io import wavfile
from scipy.signal import stft, istft
from scipy.fft import fft, ifft
from f_ssbt import ssbt, issbt, rft, irft
import gen_palette as gp
import soundfile as sf
import matplotlib.pyplot as plt
import sys
from multiprocessing import Pool

np.set_printoptions(suppress=True, precision=6, threshold=sys.maxsize)


def sim_parser(comb):
    simulation(freq_mode=comb[0], signal_mode='load', n_per_seg=comb[1])


def simulation(freq_mode: str = 'stft', signal_mode='random', n_per_seg=32):
    print('Stt:', freq_mode.upper(), n_per_seg)
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
    
    variables = scipy.io.loadmat('io_input/variables.mat')
    
    dx = variables['delta_x'].item()
    c = variables['c'].item()
    fs = variables['fs'].item()
    n_sensors = variables['n_sensors'].item()
    len_rir = variables['n'].item()
    
    global n_bins, F_lk_star, geft
    match freq_mode:
        case 'nssbt' | 'tssbt':
            n_bins = n_per_seg
            geft = ssbt
        
        case 'stft':
            n_bins = (1 + n_per_seg) // 2 + 1
            geft = stft
        
    n_bins_star = (1 + n_per_seg) // 2 + 1
    n_win_rir = int(np.ceil(2 * len_rir / n_per_seg + 1))
    sym_freqs = np.linspace(0, fs, n_per_seg + 1)[:n_bins_star]
    dist_fil = 25
    win_p_fil = 25  # INFO: If "win_p_fil = -1", uses whole signal
    m_ref = 0
    array = Array((1, n_sensors), 'Array')
    array.init_metrics(sym_freqs.size)  # Initializes metrics
    
    epsilon = 1e-12
    iSIR = 5
    iSNR = 30
    
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
    
    h_n = np.loadtxt('io_input/rir_dx_.csv', delimiter=',')
    g_n = np.loadtxt('io_input/rir_v2_.csv', delimiter=',')
    
    global x_n, v_n, r_n
    match signal_mode:
        case 'load':
            x_n, samplerate = sf.read('io_input/audio_speech_male_r.flac')
            x_n = resample(x_n, samplerate, fs)
            babble = scipy.io.loadmat('io_input/babble.mat')
            v_n, samplerate = babble['babble'].reshape(-1), babble['fs']
            v_n = resample(v_n, samplerate, fs)
            r_n, samplerate = sf.read('io_input/audio_noise_wgn.flac')
            r_n = resample(r_n, samplerate, fs).reshape(-1, 1)
            nr_n = []
            for m in range(n_sensors):
                nr_n.append(np.roll(r_n, int(m / n_sensors * r_n.size)))
            r_n = np.hstack(nr_n)
        
        case 'random' | _:
            len_x = 100000
            x_n = np.random.rand(len_x)
            v_n = np.random.rand(2 * len_x)
            r_n = np.random.rand(2 * len_x, n_sensors)
    
    """
        ------------------------------
        - Relative-impulse responses -
        ------------------------------
    """
    # Vars: Sources and signals (time)
    #       H_f         : Speech signal's RFR, for each sensor {matrix} [scalar]
    #       G_f         : Interfering signal's RFR, for each sensor {matrix} [scalar]
    #       B_f         : Speech signal's relative RFR, for each sensor {matrix} [scalar]
    #       C_f         : Interfering signal's relative RFR, for each sensor {matrix} [scalar]
    #       b_n         : Speech signal's relative RIR, for each sensor {matrix} [scalar]
    #       c_n         : Interfering signal's relative RIR, for each sensor {matrix} [scalar]
    #       x1_n        : Ref-sensor's desired signal {vector} [scalar]
    #       v1_n        : Ref-sensor's interfering signal {vector} [scalar]
    #       l_des_win   : Window in which desired signal starts [scalar]
    
    H_f = np.empty_like(h_n, dtype=complex)
    G_f = np.empty_like(g_n, dtype=complex)
    for m in range(n_sensors):
        H_f[m, :] = fft(h_n[m, :])
        G_f[m, :] = fft(g_n[m, :])
    
    # Info: Correctly deconvolving the relative impulse repsonses
    idx_max_h0_n = np.where(np.abs(h_n[m_ref, :]) == np.abs(np.amax(h_n[m_ref, :])))[0][0]
    # h0_n = np.zeros_like(h_n[m_ref, :])
    # h0_n[:-idx_max_h0_n] = h_n[m_ref, idx_max_h0_n:]
    h0_n = np.roll(h_n[m_ref, :], -idx_max_h0_n)
    H0_f = fft(h0_n)
    
    g0_n = np.roll(g_n[m_ref, :], -idx_max_h0_n)
    G0_f = fft(g0_n)
    
    B_f = H_f / H0_f
    C_f = G_f / G0_f
    
    b_n = np.empty_like(B_f, dtype=float)
    c_n = np.empty_like(C_f, dtype=float)
    for m in range(n_sensors):
        b_n[m, :] = np.real(ifft(B_f[m, :]))
        c_n[m, :] = np.real(ifft(C_f[m, :]))
    
    x1_n = np.convolve(h_n[m_ref, :], x_n, mode='full')
    v1_n = np.convolve(g_n[m_ref, :], v_n, mode='full')
    
    # INFO: Array-fixing, so that the desired signal RIR's max. value is at the start of a FT window
    idx_max_b_n = np.where(np.abs(b_n[m_ref, :]) == np.abs(np.amax(b_n[m_ref, :])))[0][0]
    new_idx_max_b_n = int(np.ceil(idx_max_b_n / (n_per_seg // 2)) * (n_per_seg // 2))
    b_n = np.hstack([np.zeros([n_sensors, new_idx_max_b_n - idx_max_b_n]), b_n])
    c_n = np.hstack([np.zeros([n_sensors, new_idx_max_b_n - idx_max_b_n]), c_n])
    
    l_des_win = new_idx_max_b_n // (n_per_seg // 2)
    
    len_rir = b_n.shape[1]
    n_win_rir = int(np.ceil(2 * len_rir / n_per_seg + 1))
    n_win_N = int(np.ceil(2 * r_n.shape[0] / n_per_seg + 1))
    
    """
        -------------------------------
        - Time-Freq. domain variables -
        -------------------------------
    """
    # Vars: Input sources and signals (freq.)
    #       B_lk        : Speech signal's GEFT, for each sensor {tensor} [scalar]
    #       C_lk        : Interfering signal's GEFT, for each sensor {tensor} [scalar]
    #       P_lk        : Undesired speech signal's GEFT, for each sensor {tensor} [scalar]
    #       dx_lk       : Speech signal's GEFT for main window, for each sensor {tensor} [scalar]
    #       dx_k        : Speech signal's GEFT for only main window, for each sensor {matrix} [scalar]
    #       X1_lk       : Speech signal's GEFT, for each sensor {tensor} [scalar]
    #       V1_lk       : Undesired signal's FT GEFT, for each sensor {tensor} [scalar]
    #       N_lk        : Noise signal's GEFT, for each sensor {tensor} [scalar]
    #       n_win_X1    : Number of windows of X_lk [scalar]
    #       n_win_V1    : Number of windows of V_lk [scalar]
    #       n_win_N     : Number of windows of N_lk [scalar]
    
    B_lk = np.empty((n_bins, n_win_rir, n_sensors), dtype=complex)
    C_lk = np.empty((n_bins, n_win_rir, n_sensors), dtype=complex)
    N_lk = np.empty((n_bins, n_win_N, n_sensors), dtype=complex)
    for idx in range(n_sensors):
        _, _, Bm_lk = geft(b_n[idx, :], fs, nperseg=n_per_seg)
        B_lk[:, :, idx] = Bm_lk
        
        _, _, Cm_lk = geft(c_n[idx, :], fs, nperseg=n_per_seg)
        C_lk[:, :, idx] = Cm_lk
        
        _, _, Rm_lk = geft(r_n[:, idx], fs, nperseg=n_per_seg)
        N_lk[:, :, idx] = Rm_lk
    
    P_lk = np.copy(B_lk)
    P_lk[:, l_des_win, :] = 0
    dx_lk = B_lk - P_lk
    dx_k = B_lk[:, l_des_win, :]
    
    _, _, X1_lk = geft(x1_n, fs, nperseg=n_per_seg)
    _, _, V1_lk = geft(v1_n, fs, nperseg=n_per_seg)
    
    n_win_X1 = X1_lk.shape[1]
    n_win_V1 = V1_lk.shape[1]
    n_win_N = N_lk.shape[1]
    
    # Vars: Input sources and signals (freq.)
    #       S_lk        : Speech signal's GEFT for each sensor {tensor} [scalar]
    #       W_lk        : Observed signal's GEFT for each sensor {tensor} [scalar]
    #       Y_lk        : Observed signal's GEFT for each sensor {tensor} [scalar]
    #       n_win_Y     : Number of windows of Y_lk [scalar]
    
    n_win_Y = n_win_rir + n_win_X1 - 1
    
    if max(n_win_V1, n_win_N) < n_win_X1 + n_win_rir - 1:
        raise AssertionError('Noise signals too short.')
    
    S_lk = np.empty((n_bins, n_win_rir + n_win_X1 - 1, n_sensors), dtype=complex)
    W_lk = np.empty((n_bins, n_win_rir + n_win_X1 - 1, n_sensors), dtype=complex)
    Y_lk = np.empty((n_bins, n_win_rir + n_win_X1 - 1, n_sensors), dtype=complex)
    for m in range(n_sensors):
        for k_idx in range(n_bins):
            # INFO: CTF convolutions and signal-length correction
            S = np.convolve(dx_lk[k_idx, :, m], X1_lk[k_idx, :], mode='full')[:n_win_Y]
            U = np.convolve(P_lk[k_idx, :, m], X1_lk[k_idx, :], mode='full')[:n_win_Y]
            I = np.convolve(C_lk[k_idx, :, m], V1_lk[k_idx, :], mode='full')[:n_win_Y]
            R = (N_lk[k_idx, :, m])[:n_win_Y]
            
            # INFO: Variance and SIR/SNR calculations
            var_S = np.var(S)
            S = S / np.sqrt(var_S + epsilon)
            U = U / np.sqrt(var_S + epsilon)
            I = I / np.sqrt(np.var(I) + epsilon) / np.sqrt(10 ** (iSIR / 10))
            R = R / np.sqrt(np.var(R) + epsilon) / np.sqrt(10 ** (iSNR / 10))
            W = U + I + R
            
            # INFO: Calculating desired, undesired, and observed signals
            S_lk[k_idx, :, m] = S
            W_lk[k_idx, :, m] = W
            Y_lk[k_idx, :, m] = S + W
    
    # Vars:
    #   "_star" variables are the "true" variable, using STFT instead of GEFT
    #       B_lk_star   : Speech signal's STFT, for each sensor {tensor} [scalar]
    #       C_lk_star   : Interfering signal's STFT, for each sensor {tensor} [scalar]
    #       P_lk_star   : Undesired speech signal's STFT, for each sensor {tensor} [scalar]
    #       dx_lk_star  : Speech signal's STFT for main window, for each sensor {tensor} [scalar]
    #       dx_k_star   : Speech signal's STFT for only main window, for each sensor {matrix} [scalar]
    #       X1_lk_star  : Speech signal's STFT, for each sensor {tensor} [scalar]
    #       V1_lk_star  : Undesired signal's FT STFT, for each sensor {tensor} [scalar]
    #       N_lk_star   : Noise signal's STFT, for each sensor {tensor} [scalar]
    
    B_lk_star = np.empty((n_bins_star, n_win_rir, n_sensors), dtype=complex)
    C_lk_star = np.empty((n_bins_star, n_win_rir, n_sensors), dtype=complex)
    N_lk_star = np.empty((n_bins_star, n_win_N, n_sensors), dtype=complex)
    for idx in range(n_sensors):
        _, _, Bm_lk_star = stft(b_n[idx, :], fs, nperseg=n_per_seg)
        B_lk_star[:, :, idx] = Bm_lk_star
        
        _, _, Cm_lk_star = stft(c_n[idx, :], fs, nperseg=n_per_seg)
        C_lk_star[:, :, idx] = Cm_lk_star
        
        _, _, Rm_lk_star = stft(r_n[:, idx], fs, nperseg=n_per_seg)
        N_lk_star[:, :, idx] = Rm_lk_star
    
    P_lk_star = np.copy(B_lk_star)
    P_lk_star[:, l_des_win, :] = 0
    dx_lk_star = B_lk_star - P_lk_star
    dx_k_star = B_lk_star[:, l_des_win, :]
    
    _, _, X1_lk_star = stft(x1_n, fs, nperseg=n_per_seg)
    _, _, V1_lk_star = stft(v1_n, fs, nperseg=n_per_seg)
    
    # Vars: Input sources and signals (freq.)
    #       S_lk_star   : Speech signal's STFT for each sensor {tensor} [scalar]
    #       W_lk_star   : Observed signal's STFT for each sensor {tensor} [scalar]
    #       Y_lk_star   : Observed signal's STFT for each sensor {tensor} [scalar]
    
    S_lk_star = np.empty((n_bins_star, n_win_rir + n_win_X1 - 1, n_sensors), dtype=complex)
    W_lk_star = np.empty((n_bins_star, n_win_rir + n_win_X1 - 1, n_sensors), dtype=complex)
    Y_lk_star = np.empty((n_bins_star, n_win_rir + n_win_X1 - 1, n_sensors), dtype=complex)
    for m in range(n_sensors):
        for k_idx in range(n_bins_star):
            # INFO: CTF convolutions and signal-length correction
            S = np.convolve(dx_lk_star[k_idx, :, m], X1_lk_star[k_idx, :], mode='full')[:n_win_Y]
            U = np.convolve(P_lk_star[k_idx, :, m], X1_lk_star[k_idx, :], mode='full')[:n_win_Y]
            I = np.convolve(C_lk_star[k_idx, :, m], V1_lk_star[k_idx, :], mode='full')[:n_win_Y]
            R = (N_lk_star[k_idx, :, m])[:n_win_Y]
            
            # INFO: Variance and SIR/SNR calculations
            var_S = np.var(S)
            U = U / np.sqrt(var_S + epsilon)
            S = S / np.sqrt(var_S + epsilon)
            I = I / np.sqrt(np.var(I) + epsilon) / np.sqrt(10 ** (iSIR / 10))
            R = R / np.sqrt(np.var(R) + epsilon) / np.sqrt(10 ** (iSNR / 10))
            W = U + I + R
            
            # INFO: Calculating desired, undesired, and observed signals
            S_lk_star[k_idx, :, m] = S
            W_lk_star[k_idx, :, m] = W
            Y_lk_star[k_idx, :, m] = S + W
    
    """
        -------------
        - Filtering -
        -------------
    """
    
    # Vars: Sources and signals (freq.)
    #       F_lk        : Beamforming filter {tensor} [scalar]
    #       n_win_F     : Number of windows of F_lk [scalar]
    #       Corr_Y      : Correlation matrix of Y_lk, for current window and bin {matrix} [scalar]
    #       F_lk_star   : Beamforming filter asserted to STFT domain {tensor} [scalar]
    #       Sf_lk_star  : Filtered S_lk {matrix} [scalar]
    #       Wf_lk_star  : Filtered W_lk {matrix} [scalar]
    #       Yf_lk_star  : Filtered Y_lk {matrix} [scalar]   - Z_lk ≡ Yf_lk
    
    if dist_fil == -1:
        dist_fil = n_win_Y
        win_p_fil = n_win_Y
    
    F_lk = np.empty((n_bins, int(np.ceil(n_win_Y / dist_fil)), n_sensors), dtype=complex)
    n_win_F = F_lk.shape[1]
    arr_delay = np.zeros([n_sensors, 2 * n_sensors], dtype = complex)
    for m in range(n_sensors):
        arr_delay[m, m] = np.exp(1j * 3 * PI / 4) / np.sqrt(2)
        arr_delay[m, n_sensors + m] = np.exp(-1j * 3 * PI / 4) / np.sqrt(2)
    id = np.array([[1], [0]])
    for k_idx in range(n_bins):
        D = dx_k[k_idx, :]
        D = D.reshape(-1, 1)
        for l_idx in range(n_win_F):
            # INFO: Separating necessary windows of Y_lk, and calculating coherence matrix
            idx_stt = max(0, (l_idx + 1) * dist_fil - win_p_fil)
            idx_end = min((l_idx + 1) * dist_fil, n_win_Y)
            sig = Y_lk
            match freq_mode:
                case 'stft' | 'nssbt':
                    O = sig[k_idx, idx_stt:idx_end, :]
                    Corr_O = np.empty([n_sensors, n_sensors], dtype=complex)
                    for idx_i in range(n_sensors):
                        for idx_j in range(idx_i, n_sensors):
                            Oi = O[:, idx_i].reshape(-1, 1)
                            Oj = O[:, idx_j].reshape(-1, 1)
                            Corr_O[idx_i, idx_j] = (he(Oi) @ Oj).item()
                            Corr_O[idx_j, idx_i] = np.conj(Corr_O[idx_i, idx_j])
                    try:
                        iCorr_O = inv(Corr_O)
                        F_lk[k_idx, l_idx, :] = ((iCorr_O @ D) / (he(D) @ iCorr_O @ D + epsilon)).reshape(n_sensors)
                    except np.linalg.LinAlgError:
                        F_lk[k_idx, l_idx, :] = 0
                
                case 'tssbt':
                    if k_idx >= n_bins_star:
                        continue
                    o1 = sig[k_idx, idx_stt:idx_end, :]
                    if k_idx == 0 or (k_idx == n_bins_star-1 and n_bins/2 == n_bins//2):
                        # Info: Breaks if k=0 or k=K/2 with K even, so this shall be considered.
                        Corr_O = np.empty((n_sensors, n_sensors), dtype=float)
                        for idx_i in range(n_sensors):
                            for idx_j in range(n_sensors):
                                Oi = o1[:, idx_i].reshape(-1, 1)
                                Oj = o1[:, idx_j].reshape(-1, 1)
                                Corr_O[idx_i, idx_j] = np.real(tr(Oi) @ Oj).item()
                                Corr_O[idx_j, idx_i] = Corr_O[idx_i, idx_j]
                                
                        Dx = (dx_k_star[k_idx, :]).reshape(-1, 1)
                        Q = np.hstack([np.real(Dx), np.imag(Dx)])
                        try:
                            iCorr_O = inv(Corr_O + np.eye(n_sensors)*epsilon)
                            Fm_lk = iCorr_O @ Q @ inv(tr(Q) @ iCorr_O @ Q)
                            Fm_lk = Fm_lk @ id
                        except np.linalg.LinAlgError:
                            Fm_lk = np.zeros((n_sensors, 1))
                        F_lk[k_idx, l_idx, :] = Fm_lk.reshape(n_sensors)
                    else:
                        o2 = sig[-k_idx, idx_stt:idx_end, :]
                        Corr_O = np.empty((2*n_sensors, 2*n_sensors), dtype=float)
                        for idx_i in range(2*n_sensors):
                            for idx_j in range(2*n_sensors):
                                if idx_i < n_sensors:
                                    Oi = o1[:, idx_i].reshape(-1, 1)
                                else:
                                    Oi = o2[:, idx_i-n_sensors].reshape(-1, 1)
                                if idx_j < n_sensors:
                                    Oj = o1[:, idx_j].reshape(-1, 1)
                                else:
                                    Oj = o2[:, idx_j-n_sensors].reshape(-1, 1)
                                Corr_O[idx_i, idx_j] = np.real(tr(Oi) @ Oj).item()
                                Corr_O[idx_j, idx_i] = Corr_O[idx_i, idx_j]
                    
                        Dx = he(arr_delay) @ ((dx_k_star[k_idx, :]).reshape(-1, 1))
                        Q = np.hstack([np.real(Dx), np.imag(Dx)])
                        try:
                            iCorr_O = inv(Corr_O)
                            Fm_lk = iCorr_O @ Q @ inv(tr(Q) @ iCorr_O @ Q)
                            Fm_lk = Fm_lk @ id
                        except np.linalg.LinAlgError:
                            Fm_lk = np.zeros((2*n_sensors, 1))
                        F_lk[k_idx, l_idx, :] = Fm_lk[:n_sensors, 0].reshape(n_sensors)
                        F_lk[-k_idx, l_idx, :] = Fm_lk[n_sensors:, 0].reshape(n_sensors)
    
    # Info: Assuring filter is in STFT
    match freq_mode:
        case 'stft':
            # Info: F_lk_star.shape = [n_bins_star, n_win_F, n_sensors]
            F_lk_star = F_lk
        case 'nssbt':
            F_lk_star = np.empty((n_bins_star, n_win_F, n_sensors), dtype=complex)
            for l_idx in range(n_win_F):
                for m in range(n_sensors):
                    fm_ln = irft(F_lk[:, l_idx, m])
                    Fm_lk = fft(fm_ln)[:Y_lk_star.shape[0]]
                    F_lk_star[:, l_idx, m] = Fm_lk
        case 'tssbt':
            F_lk_star = np.empty((n_bins_star, n_win_F, n_sensors), dtype=complex)
            for l_idx in range(n_win_F):
                for k_idx in range(n_bins_star):
                    Fs = np.vstack([F_lk[k_idx, l_idx, :].reshape(-1, 1), F_lk[-k_idx, l_idx, :].reshape(-1, 1)])
                    F_lk_star[k_idx, l_idx, :] = (arr_delay @ Fs).reshape(-1)
    
    Sf_lk_star = np.empty((n_bins_star, n_win_Y), dtype=complex)
    Wf_lk_star = np.empty((n_bins_star, n_win_Y), dtype=complex)
    Yf_lk_star = np.empty((n_bins_star, n_win_Y), dtype=complex)
    
    for k_idx in range(n_bins_star):
        for l_idx in range(n_win_Y):
            F = F_lk_star[k_idx, l_idx // dist_fil, :].reshape(-1, 1)
            S = S_lk_star[k_idx, l_idx, :].reshape(-1, 1)
            W = W_lk_star[k_idx, l_idx, :].reshape(-1, 1)
            Y = Y_lk_star[k_idx, l_idx, :].reshape(-1, 1)
            
            Sf_lk_star[k_idx, l_idx] = (he(F) @ S).item()
            Wf_lk_star[k_idx, l_idx] = (he(F) @ W).item()
            Yf_lk_star[k_idx, l_idx] = (he(F) @ Y).item()
    
    """
        -----------
        - Metrics -
        -----------
    """
    
    # Vars: Sources and signals (freq.)
    #       gSINR_lk    : Narrowband gain in SNR per-window [scalar, dB]
    #       gSINR_k     : Narrowband window-average gain in SNR [scalar, dB]
    #       dsdi_lk     : Narrowband desired-signal distortion index [scalar]
    
    gSINR_lk = np.empty((n_bins_star, n_win_F), dtype=float)
    DSDI_lk = np.empty((n_bins_star, n_win_F), dtype=float)
    gSINR_k = np.empty((n_bins_star,), dtype=float)
    DSDI_k = np.empty((n_bins_star,), dtype=float)
    
    for k_idx in range(n_bins_star):
        for l_idx in range(n_win_F):
            idx_stt = max(0, (l_idx + 1) * dist_fil - win_p_fil)
            idx_end = min((l_idx + 1) * dist_fil, n_win_Y)
            S = S_lk_star[k_idx, idx_stt:idx_end].reshape(-1, 1)
            W = W_lk_star[k_idx, idx_stt:idx_end].reshape(-1, 1)
            
            Sf = Sf_lk_star[k_idx, idx_stt:idx_end].reshape(-1, 1)
            Wf = Wf_lk_star[k_idx, idx_stt:idx_end].reshape(-1, 1)
            
            var_S = np.var(S)
            var_W = np.var(W)
            
            var_Sf = np.var(Sf)
            var_Wf = np.var(Wf)
            
            iSINR = (var_S + epsilon) / (var_W + epsilon)
            oSINR = (var_Sf + epsilon) / (var_Wf + epsilon)
            gSINR_lk[k_idx, l_idx] = np.real((oSINR + epsilon) / (iSINR + epsilon))
            
            F = F_lk_star[k_idx, l_idx // dist_fil, :].reshape(-1, 1)
            D = dx_k_star[k_idx, :].reshape(-1, 1)
            DSDI_lk[k_idx, l_idx] = (np.abs(he(F) @ D - 1) ** 2).item()
    
    for k_idx in range(n_bins_star):
        gSINR_k[k_idx] = dB(np.mean(reject_outliers(gSINR_lk[k_idx, :])))
        DSDI_k[k_idx] = np.mean(reject_outliers(DSDI_lk[k_idx, :]))
    gSINR_lk = dB(gSINR_lk)
    
    """
        ---------------
        - Export data -
        ---------------
    """
    
    exp_gSINR_lk = ['freq, win, val']
    exp_gSINR_k = ['freq, val']
    exp_DSDI_lk = ['freq, win, val']
    exp_DSDI_k = ['freq, val']
    
    sym_freqs = sym_freqs / 1000
    
    k_stt, k_end = 1, n_bins_star-1
    # k_stt, k_end = 0, n_bins_star
    
    for k_idx in range(k_stt, k_end):
        exp_gSINR_k.append(','.join([str(sym_freqs[k_idx]), str(gSINR_k[k_idx])]))
        exp_DSDI_k.append(','.join([str(sym_freqs[k_idx]), str(DSDI_k[k_idx])]))
        
        for l_idx in range(n_win_F):
            t = l_idx * dist_fil * n_bins_star / fs
            exp_gSINR_lk.append(','.join([str(sym_freqs[k_idx]), str(t), str(gSINR_lk[k_idx, l_idx])]))
            exp_DSDI_lk.append(','.join([str(sym_freqs[k_idx]), str(t), str(DSDI_lk[k_idx, l_idx])]))
    
    exp_gSINR_k = '\n'.join(exp_gSINR_k)
    exp_gSINR_lk = '\n'.join(exp_gSINR_lk)
    exp_DSDI_lk = '\n'.join(exp_DSDI_lk)
    exp_DSDI_k = '\n'.join(exp_DSDI_k)
    
    freq_mode = freq_mode.upper()
    filename = '_' + freq_mode + '_' + str(n_per_seg)
    folder = 'io_output/' + freq_mode + '/'
    if not os.path.isdir('io_output/'):
        os.mkdir('io_output/')
    if not os.path.isdir(folder):
        os.mkdir(folder)
    with open(folder + 'gain_SINR_k' + filename + '.csv', 'w') as f:
        f.write(exp_gSINR_k)
        f.close()
    with open(folder + 'gain_SINR_lk' + filename + '.csv', 'w') as f:
        f.write(exp_gSINR_lk)
        f.close()
    with open(folder + 'DSDI_k' + filename + '.csv', 'w') as f:
        f.write(exp_DSDI_k)
        f.close()
    with open(folder + 'DSDI_lk' + filename + '.csv', 'w') as f:
        f.write(exp_DSDI_lk)
        f.close()
    
    _, yf_n = istft(Yf_lk_star, fs)
    yf_n = 0.9 * yf_n / np.amax(yf_n)
    wavfile.write('io_output/audios/yf_' + freq_mode + '_' + str(n_per_seg) + '.wav', fs, yf_n)
    
    _, y_n = istft(Y_lk_star[:, :, 0], fs)
    y_n = 0.9 * y_n / np.amax(y_n)
    wavfile.write('io_output/audios/y1_unfiltered.wav', fs, y_n)
    
    _, x1_n = istft(X1_lk_star, fs)
    x1_n = 0.9 * x1_n / np.amax(x1_n)
    wavfile.write('io_output/audios/x1_unfiltered.wav', fs, x1_n)
    
    """
        --------------------
        - Export aux files -
        --------------------
    """
    
    t_min = 0
    t_max = n_win_Y * n_bins_star / fs
    f_step = sym_freqs[1] - sym_freqs[0]
    f_min = sym_freqs[k_stt] - f_step/2
    f_max = sym_freqs[k_end-1] + f_step/2
    
    mesh_cols = r'\def\meshcols{{{}}}'.format(n_win_F)
    mesh_rows = r'\def\meshrows{{{}}}'.format(n_bins_star-2)
    t_min = r'\def\tmin{{{}}}'.format(t_min)
    t_max = r'\def\tmax{{{}}}'.format(t_max)
    f_min = r'\def\fmin{{{}}}'.format(f_min)
    f_max = r'\def\fmax{{{}}}'.format(f_max)
    data = [mesh_cols, mesh_rows, t_min, t_max, f_min, f_max]
    data_defs = '\n'.join(data)
    
    with open('io_output/' + 'aux_data_' + str(n_per_seg) + '.tex', 'w') as f:
        f.write(data_defs)
        f.close()
    
    colors = gp.gen_palette(80, 60, ['A', 'B', 'C', 'D', 'E', 'F'], 345)
    color_defs = '\n'.join(colors)
    
    with open('io_output/' + 'colors_' + str(len(colors)) + '.tex', 'w') as f:
        f.write(color_defs)
        f.close()
    
    print('End:', freq_mode, n_per_seg)
    return None


def main():
    freqmodes = [
        # 'stft',
        # 'nssbt',
        'tssbt'
    ]
    
    npersegs = [
        32,
        64,
    ]
    
    combs = [(freqmode, nperseg) for freqmode in freqmodes for nperseg in npersegs]
    ncombs = min(len(combs), 4)
    parallel = True
    if parallel:
        with Pool(ncombs) as p:
            p.map(sim_parser, combs)
    else:
        for comb in combs:
            sim_parser(comb)


if __name__ == '__main__':
    main()
