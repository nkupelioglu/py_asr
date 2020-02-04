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

class py_asr:
    def __init__(self):
        pass

    def asr_make_calibration(self, signal, fs, z_param=(-3.5, 5.5), n_windows=100):
        n_channels = signal.shape[0]
        n_timepoints = signal.shape[1]
        n_total_possible_windows = (n_timepoints / fs) - 1
        n_clean_windows = 0
        n_dirty_windows = 0
        calib_window = None
        for i in range(int(n_total_possible_windows)):
            window = signal[:, i * fs:i * fs + fs]
            rms_scores = []
            for j in range(n_channels):
                rms = np.sqrt(np.mean(window[j, :] ** 2))
                rms_scores.append(rms)
            z_scores = stats.zscore(np.array(rms_scores))
            if z_scores.max() < z_param[1] and z_scores.min() > z_param[0] and n_clean_windows < n_windows:
                if n_clean_windows == 0:
                    calib_window = window
                else:
                    calib_window = np.concatenate((calib_window, window), axis=1)
                n_clean_windows += 1
            else:
                n_dirty_windows += 1
        # print("Clean: " + str(n_clean_windows))
        # print("Dirty: " + str(n_dirty_windows))
        return calib_window

    def asr_rejection_criteria(self, x_c, fs, k=2):
        # Calculation of covariance matrix.
        m_c = np.cov(x_c)
        # Eigenvalue decomposition of the covariance matrix.
        d_c, v_c = np.linalg.eig(m_c)
        # Projection of the data into component space.
        y_c = np.dot(v_c.T, x_c)
        # Calculation of mean and std.dev of RMS values accross 0.5-second windows for each component i.
        n_total_windows = x_c.shape[1] / (fs / 2)
        n_channels = x_c.shape[0]
        window_size = int(fs / 2)
        mu_c = []
        std_c = []
        T = []
        for j in range(n_channels):
            rms_scores = []
            for i in range(int(n_total_windows)):
                window = y_c[j, i * (window_size):i * (window_size) + (window_size)]
                rms = np.sqrt(np.mean(window ** 2))
                rms_scores.append(rms)
            mu, std = norm.fit(rms_scores)
            mu_c.append(mu)
            std_c.append(std)
        for i in range(len(mu_c)):
            T.append(mu_c[i] + k * std_c[i])
        return m_c, d_c, v_c, T

    def asr_clean(self, signal, fs, x_c, m_c, v_c, d_c, T, win_len=0.5, win_step=0.25):
        keeps = []
        out_signal = signal
        n_channels = signal.shape[0]
        n_timepoints = signal.shape[1]
        n_total_possible_windows = (n_timepoints / fs * win_step) - 1
        win_len_points = fs * win_len
        for i in range(int(n_total_possible_windows)):
            window = signal[:, i * int((fs * win_step)):i * int((fs * win_step)) + int((fs * win_len))]
            cov_window = np.cov(window)
            d_window, v_window = np.linalg.eig(cov_window)
            keep = []
            for j in range(len(T)):
                d_window_j = np.real(d_window[j])
                sum_dot_j = np.real(np.sum((np.dot(np.dot(T[j], v_c.T), v_window[j])) ** 2))
                if d_window_j < sum_dot_j:
                    keep.append(1)
                else:
                    keep.append(0)
            if np.sum(keep) < n_channels:
                # Reconstruct window
                # R = real(M*pinv(bsxfun(@times,keep',V'*M))*V');
                keep = np.array(keep)
                v_trunk_m_c = pinv(np.multiply(keep.T, np.dot(v_window.T, m_c)))
                reconsturction_matrix = np.real(np.dot(np.dot(m_c, v_trunk_m_c), v_window.T))
                # reconsturction_matrix = np.real(np.dot(np.dot(m_c,np.dot(keep.T,m_c).conj().T),v_window.T))
                updated_window = np.dot(reconsturction_matrix, window)
                out_signal[:, i * int((fs * win_step)):i * int((fs * win_step)) + int((fs * win_len))] = updated_window
            # print('asd')
            # print(keep)
        return out_signal