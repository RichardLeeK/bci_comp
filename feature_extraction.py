import scipy.signal as sgn
import numpy as np
import scipy as sp
import pywt
# from mne.time_frequency import tfr_stockwell

def csp(x, y):
    # only for binary
    x1 = []; x2 = []
    n_channels = x.shape[-1]
    for i in range(len(y)):
        if y[i] == 0:
            x1.append(x[i])
        else:
            x2.append(x[i])
    x1 = np.array(x1).reshape(-1, n_channels)
    x2 = np.array(x2).reshape(-1, n_channels)
    c1 = np.cov(x1.transpose())
    c2 = np.cov(x2.transpose())
    d, v = sp.linalg.eig(c1-c2, c1+c2)
    d = d.real
    idx = np.argsort(d)
    idx = idx[::-1]
    d = d.take(idx)
    v = v.take(idx, axis=1)
    a = sp.linalg.inv(v).transpose()
    return v, a, d

def stft_fe(x, fs=250):
    # short time Fourier transform
    ci = x.shape[0] # the number of trials
    cj = x.shape[-1] # the number of channels
    res = []
    for i in range(ci):
        cur_trial  = []
        for j in range(cj):
            _, _, z = sgn.stft(x[i,:,j], fs, nperseg=100)
            cur_trial.append(z[:12,:])
        res.append(np.array(cur_trial).T)
    return np.array(res)

# def stock_well_fe(x):
#     from mne.datasets import somato

#     power, itc = tfr_stockwell(x, fmin=8., fmax=30., return_itc=True)
#     pass

def wavelet_1dc_fe(x):
    # continuous one dimensional wavelet transform
    ci = x.shape[0]; cj = x.shape[-1]; res = []
    for i in range(ci):
        cur_trial = []
        for j in range(cj):
            coef, _ = pywt.cwt(x[i,:,j], np.arange(8, 30), 'morl', sampling_period=0.004)
            cur_trial.append(coef)
        res.append(np.array(cur_trial).T)
    return np.array(res)


if __name__ == '__main__':
    raw = np.load('data/comp_iva/test.npz', allow_pickle=True)
    x = raw['data'][0]['x']
    y = raw['data'][0]['y']
    # a, b, c = stft_fe(x)
    wavelet_1dc_fe(x)