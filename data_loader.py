from matplotlib.collections import LineCollection
from scipy.io import loadmat
import numpy as np
data_folder = 'D:\\16_bci_competition_IV_1\\data\\'
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
from numpy.linalg import multi_dot
from numpy.linalg import pinv
import py_asr as pyasr

from scipy.signal import butter, lfilter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def load_bci_comp(path):
    bci_file = loadmat(path)
    event_lat = bci_file['mrk'][0][0][0][0]
    event_lab = bci_file['mrk'][0][0][1][0]
    chan_names = []
    for i in range(len(bci_file['nfo'][0][0][2][0])):
        chan_names.append(bci_file['nfo'][0][0][2][0][i][0])
    chan_names = np.array(chan_names)
    chan_x = []
    for i in range(len(bci_file['nfo'][0][0][3])):
        chan_x.append(bci_file['nfo'][0][0][3][i][0])
    chan_x = np.array(chan_x)
    chan_y = []
    for i in range(len(bci_file['nfo'][0][0][4])):
        chan_y.append(bci_file['nfo'][0][0][4][i][0])
    chan_y = np.array(chan_y)
    signal = bci_file['cnt'].T
    fs = bci_file['nfo'][0][0][0][0][0]
    ret_data = {'signal': signal, 'chan_names': chan_names, 'chan_x': chan_x, 'chan_y': chan_y, 'event_labels': event_lab, 'event_latencies': event_lat, 'fs': fs}
    return ret_data

def filter_all_channels(signal, fs):
    channel = []
    for i in range(signal.shape[0]):
        filtered_signal = butter_bandpass_filter(signal[i], 1, 40, data['fs'], order=2)
        channel.append(filtered_signal)
    return np.array(channel)

bci_competition_files = [f for f in listdir(data_folder) if isfile(join(data_folder, f))]

def plot_channels(filtered_signal):
    f_signal_sample = filtered_signal[:, 0:1000]
    # Plot the EEG
    fig = plt.figure("EEG Signal")
    ticklocs = []
    ax2 = plt.axes()
    ax2.set_xlim(0, 10)
    ax2.set_xticks(np.arange(10))
    dmin = f_signal_sample.min()
    dmax = f_signal_sample.max()
    dr = (dmax - dmin)  # Crowd them a bit.
    y0 = dmin
    y1 = (f_signal_sample.shape[0] - 1) * dr + dmax
    ax2.set_ylim(y0, y1)

    segs = []

    t = 100 * np.arange(f_signal_sample.shape[1]) / f_signal_sample.shape[1]
    for i in range(f_signal_sample.shape[0]):
        segs.append(np.column_stack((t, f_signal_sample[i, :])))
        ticklocs.append(i * dr)

    offsets = np.zeros((f_signal_sample.shape[0], 2), dtype=float)
    offsets[:, 1] = ticklocs

    lines = LineCollection(segs, offsets=offsets, transOffset=None)
    ax2.add_collection(lines)

    # Set the yticks to use axes coordinates on the y axis
    ax2.set_yticks(ticklocs)
    ax2.set_yticklabels(data['chan_names'])

    ax2.set_xlabel('Time (s)')

    plt.show()


asr = pyasr.py_asr()
#for i in range(len(bci_competition_files)):
#    print(join(data_folder,bci_competition_files[i]))
#print(bci_competition_files)
bci_file = loadmat(join(data_folder,bci_competition_files[0]))
data = load_bci_comp(join(data_folder,bci_competition_files[0]))
filtered_signal = filter_all_channels(data['signal'],data['fs'])
to_clean = np.copy(filtered_signal)
x_c = asr.asr_make_calibration(filtered_signal,data['fs'])
m_c, d_c, v_c, T = asr.asr_rejection_criteria(x_c,data['fs'])
cleaned_signal = asr.asr_clean(to_clean, data['fs'], x_c, m_c, v_c, d_c, T)