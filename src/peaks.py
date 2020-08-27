from scipy.signal import argrelextrema
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook

from utils import convert_time_to_sample, convert_sample_to_time

from config import *


def get_erp_scores(epochs, channel):
    scores = epochs.average().data[channel]
    return scores


def detect_peaks(scores, start_time=200, end_time=450, draw_peak=False, order=10):
    """
    TODO should be refactored
    
    Detect local maximas of a signal between start time and end time
    
    Returns all local maximas in a specified range
    """
    max_indices,  = argrelextrema(scores, np.greater, order=order)
    
    start_index = convert_time_to_sample(start_time)
    end_index = convert_time_to_sample(end_time)
    max_indices_restricted = max_indices[(max_indices >= start_index) & (max_indices < end_index)]
    
    if draw_peak:
        time = np.linspace(-100, 700, 201)
        plt.figure(figsize=(10, 5))
        plt.plot(time, scores)
        plt.scatter(time[max_indices], scores[max_indices])
        plt.scatter(time[max_indices_restricted], scores[max_indices_restricted], s=100)

        plt.axvline(start_time)
        plt.axvline(end_time)
        plt.show()
        
    return max_indices_restricted


def detect_peak(scores, start_time=200, end_time=450, draw_peak=False, order=10):
    """
    Detect most dominant peak in a specified range
    """
    max_indices_restricted = detect_peaks(scores, start_time, end_time, draw_peak, order)
    global_max = max_indices_restricted[np.argmax(scores[max_indices_restricted])]
    return global_max


def extract_peak_features(scores, max_index, order=10):
    """
    Extract features from given peak (max_index):
    - Peak latency 
    - Difference between the peak and predecessor/successor lowest point amplitude/latency
    """
    min_indices,  = argrelextrema(scores, np.less, order=order)
    start_min_index = min_indices[min_indices < max_index].max()
    end_min_indices = min_indices[min_indices > max_index]
    end_min_index = scores.shape[0] - 1
    if end_min_indices.shape[0]:
        end_min_index = end_min_indices.min()
    
    erp_latency = convert_sample_to_time(max_index)
    erp_start_amplitude = scores[max_index] - scores[start_min_index]
    erp_end_amplitude = scores[max_index] - scores[end_min_index]
    erp_start_delay = erp_latency - convert_sample_to_time(start_min_index)
    erp_end_delay = convert_sample_to_time(end_min_index) - erp_latency
    
    return [erp_latency, erp_start_amplitude, erp_end_amplitude, erp_start_delay, erp_end_delay]


def extract_features(subject, epochs, sliding_window_size=None, channels=len(CHANNELS)):
    """
    Extract peak features for each channels of given epochs. 
    The ERP waveform will be averaged in a sliding manner by a window of specified size
    
    Returns peak features for n_channels * n_windows trials
    """
    features = []
    
    if not sliding_window_size:
        sliding_window_size = len(epochs)
    
    events_length = epochs.selection.shape[0]
    print("Averaging by {} attemps".format(events_length))
    
    sliding_epochs = epochs
    
    channels_features = []
    try:
        for channel in range(0, channels):
            scores = get_erp_scores(sliding_epochs, channel)
            max_index = detect_peak(scores, draw_peak=False, order=3)
            peak_features = extract_peak_features(scores, max_index, order=3)
            channels_features.append(peak_features + [channel, subject])
    except ValueError:
        print("value error")
        pass
    features += channels_features
    
    return features


def find_peak_sequence(peaks, max_classes=8, expected_peak=300):
    """
    TODO experimental method
    TODO description and proof
    
    Find the biggest sequence of dominant peaks through all channels around the specified region
    """
    lengths = {}
    peaks = list(sorted(peaks))
    for index, point in enumerate(peaks):
        classes = set([point[1]])
        distance = 0
        prev_point = point
        for next_point in peaks[index + 1:]:
            if next_point[1] not in classes:
                classes.add(next_point[1])
                distance += (next_point[0] - prev_point[0])
                prev_point = next_point
        
        if len(classes) == max_classes:
            lengths[point[0]] = distance # + np.abs(expected_peak - point[0])
    return lengths
