import numpy as np
from numpy import pi as PI
import scipy.signal as sps

def err(mat1, mat2, dec=2):
    mat_err = mat1 - mat2
    mat_err = fix_decimals(mat_err, dec)
    return mat_err


def fix_decimals(in_mat, dec=2):
    # Fixes decimal places
    if type(in_mat) not in (list, tuple, set):
        in_mat = [in_mat]
    mats = list(in_mat)
    for idx in range(len(mats)):
        mat = mats[idx]
        # fac = 10 ** dec
        # mat = (1 / fac) * np.around(mat * fac)
        mat = np.around(mat, decimals=dec)
        mats[idx] = mat
    mats = tuple(mats)
    return mats


def he(mat):
    # Hermitian of a matrix
    mat = mat.T
    mat = np.conj(mat)
    return mat


def tr(mat):
    mat = mat.T
    return mat


def calc_gain(h, d, Corr_V, Var_V):

    gain = Var_V * abs(he(h) @ d)**2 / (he(h) @ Corr_V @ h)
    gain = np.real(gain).item()
    if all is True:
        return gain
    else:
        return gain


def calc_wng(h, d):
    # Calculates white noise gain

    A = np.abs(d[0])**2 / (he(h) @ h)
    A = np.real(A).item()

    return A


def calc_snr(Arr, varx, ti, varis, vard, vara, f, c):
    # Calculates SNR
    CorrV = np.zeros([Arr.M, Arr.M], dtype=complex)
    for idx, t in enumerate(ti):
        d_ti = Arr.calc_sv(t, f, c, False)
        CorrV += varis[idx] * d_ti @ he(d_ti)
    CorrV += np.identity(Arr.M) * vard
    CorrV += np.ones_like(CorrV) * vara

    h = Arr.h
    d = Arr.d_td
    A = (varx * np.abs(he(h) @ d) ** 2 / (he(h) @ CorrV @ h)) / (varx / CorrV[0, 0])
    A = np.real(A).item()

    return A


def cart2pol(A):
    # Converts cartesian to polar
    Ar = np.abs(A)
    Ap = np.angle(A)
    Ap[Ap < 0] = 2 * PI + Ap[Ap < 0]

    return Ar, Ap


def pol2cart(Ar, Ap):
    # Converts polar to cartesian
    A = Ar * np.exp(1j * Ap)

    return A


def dB(A):
    # Calculates value in dB
    if type(A) in [list, tuple, set]:
        B = list(A)
    else:
        B = [A]
    C = []
    for b in B:
        c = 10 * np.log10(np.abs(b) + 1e-15)
        C.append(c)
    if type(A) in [list, tuple, set]:
        C = tuple(C)
    else:
        C = C[0]

    return C


class Array:
    # Class Array (of sensors)
    def __init__(self, M, name):
        My, Mx = M
        self.name = name
        self.Mx = Mx
        self.My = My
        self.M = Mx*My

        self.metrics = Params(gain=None, wng=None, df=None, snr=None)

    def calc_gain(self, Corr_V, Var_X, Var_V):
        gain = calc_gain(self.h, self.d_x, Corr_V, Var_X, Var_V)
        return gain

    def init_metrics(self, len_f):
        self.metrics.wng = np.zeros([len_f])
        self.metrics.df = np.zeros([len_f])
        self.metrics.snr = np.zeros([len_f])
        self.metrics.gain = np.zeros([len_f])


class Params:
    # Class Params (to group various parameters)
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def add(self, **kwds):
        self.__dict__.update(kwds)


def resample(old_signal, old_rate, new_rate):
    old_length = old_signal.size
    new_length = int(np.ceil(old_length * new_rate / old_rate))
    new_signal = sps.resample(old_signal, new_length)

    return new_signal


def reject_outliers(data, m=5):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else np.zeros(len(d))
    return data[s < m]
    