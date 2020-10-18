from scipy.io import loadmat
import numpy as np
import pandas as pd
import mne
from sklearn.preprocessing import StandardScaler

from config import *
from visualize import *


def restore_raw(session_df, events_column="Trigger", info=None):
    """
    Restore Raw object from dataframe with signal.
    The restored object will contain all EEG channels and one stim channel
    """
    channels = CHANNELS + [events_column]
    if not info:
        info = mne.create_info(ch_names=channels, sfreq=FREQUENCY)

    raw = mne.io.RawArray(session_df[channels].values.T, info)

    channel_types = {c: 'eeg' for c in channels}
    channel_types[events_column] = 'stim'

    raw.set_channel_types(channel_types)
    
    return raw


def filter_signal(raw):
    """
    Perform notch filtering for power line noise (50 Hz)
    Perform bandpass filtering of trend and high frequencies (0.5-30 Hz) 
    """
    raw.notch_filter(50)
    raw.filter(0.1, 10)


def get_events(raw, min_duration=len(TIME), events_column="Trigger"):
    """
    Get non-overlapping events in a following manner:
    - If previous event is target stim and next event overlaps previous one, then do not add next event
    - If previous event is a common stim and next event is a target stim, and there are too few of target events,
    then remove previous event and add next one
    - Otherwise, just add the next event
    
    Returns the list of non-overlapping events
    """
    
    events = mne.find_events(raw, events_column, initial_event=True)
    
    return events
#     non_overlapping_events = [events[0]]
#     positive_events = 0

#     for current_index, event in enumerate(events[1:]):
#         previous_event = non_overlapping_events[-1]
#         positive_events += previous_event[-1] == 1

#         if (previous_event[-1] == 1) and (event[0] - previous_event[0] > min_duration):
#             non_overlapping_events.append(event)

#         elif (previous_event[-1] != 1) and (event[-1] == 1) and (current_index // 2 > positive_events):
#             non_overlapping_events.pop(-1)
#             non_overlapping_events.append(event)

#         elif (previous_event[-1] != 1):
#             non_overlapping_events.append(event)

#     return np.array(non_overlapping_events)


def reject_bad_events(raw, events, channels=len(CHANNELS), duration=len(TIME)):
    """
    Removes events with abnormal mean
    
    For each event type, it checks mean and std over all the signal, 
    and then filters out those events which have mean > total_mean +- std
    at least for one channel
    """
    
    # TODO fix
    channels = raw.info["nchan"]
    
    new_events = []
    
    # Fix bad events
    
    for type in np.unique(events.T[-1]):
        indiced_total_data = [
            (index, raw[:channels, event[0]:event[0] + duration][0]) 
            for index, event in enumerate(events) if event[-1] == type
        ]
        total_data = [a for _, a in indiced_total_data]
        total_mean = np.mean(total_data)
        total_std = np.std(total_data)
        
        for index, event_data in indiced_total_data:
            bad = False
            
            for channel in range(0, channels):
                channel_event_data = event_data[channel]
                channel_event_mean = channel_event_data.mean()
                if np.abs(channel_event_mean - total_mean) >= total_std:
                    bad = True
                    break
            
            if not bad:
                new_events.append(index)
    return np.array([e for i, e in enumerate(events) if i in new_events])


def scale_signal(raw, events, events_column='Trigger'):
    """
    TODO requires additional settings for BCI III dataset
    
    The method scales the signal using sklearn StandardScaler, fitted on some baseline interval without events.
    """
    channels = CHANNELS + [events_column]
    
    channels_data = raw.copy().pick_types(eeg=True).get_data()
    # TODO use only data before events - not to loose info about amplitude
#     training_data = channels_data[:, (events[0][0] // 2):events[0][0] - 1]
    training_data = channels_data
    
    trigger_data = raw.copy().pick_types(stim=True).get_data()[0]
    
    scaler = StandardScaler()
    scaler.fit(training_data.T)
    scaled_channels_data = scaler.transform(channels_data.T).T

    np.mean(scaled_channels_data, axis=1)

    new_channels_data = np.vstack([
        scaled_channels_data,
        trigger_data[np.newaxis, :]
    ])
    
    session_df = pd.DataFrame(new_channels_data.T, columns=channels)
    
    return restore_raw(session_df, info=raw.info)


def get_epochs(raw, events):
    """
    Extract epochs from signal baselined by the interval before the event
    """
    
    raw = raw.copy()
    
    epochs = mne.Epochs(
        raw.pick_types(eeg=True), 
        events, tmin=-0.1, tmax=0.7,
        preload=True, baseline=(None, 0)
    )
    return epochs


def process_raw_signal(raw, start=30, stop=50, events_column="Trigger", draw=False):
    """
    Perform signal processing for a given dataframe:
    - Convert to Raw instance
    - Extract non-overlapping events
    - Filter signal frequencies
    - Scale signal (TODO should be fixed for BCI III)
    - Reject bad events
    EEG is visualized after each step between start and stop params
    The events will be taken from specified column
    """
    
    events = get_events(raw, events_column=events_column)
    
    if draw:
        plot_raw(raw, start=start, stop=stop)
    
    filter_signal(raw)
    
    if draw:
        plot_raw(raw, start=start, stop=stop)
    
#     raw = scale_signal(raw, events)
#     plot_raw(raw, scale=10, start=start, stop=stop)
    
#     events = reject_bad_events(raw, events)

    assert events.shape[0] > 0, "Events are empty"
    
    epochs = get_epochs(raw, events)
    
    if draw:
        plot_p300(epochs)
        
    return epochs


def process_signal(session_df, start=30, stop=50, events_column="Trigger", draw=True):
    raw = restore_raw(session_df, events_column=events_column)
    epochs = process_raw_signal(raw, start, stop, events_column, draw)
    
    return epochs
