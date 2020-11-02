from scipy.signal import butter, lfilter
import numpy as np

def bandpass_filter(data, lowcut, highcut, fs, order=5):
  nyq = 0.5 * fs
  low = lowcut / nyq
  high = highcut / nyq
  b, a = butter(order, [low, high], btype='band')
  y = lfilter(b, a, data)
  return y

def arr_bandpass_filter(data, lowcut, highcut, fs, order=5):
  y = np.array(data)
  for i in range(len(data[0])):
    cur_data = data[:,i]
    cur_y = bandpass_filter(cur_data, lowcut, highcut, fs, order)
    y[:,i] = cur_y
  return y