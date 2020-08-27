import matplotlib.pyplot as plt
from pylab import rcParams
import mne


def plot_raw(raw, start=30, stop=50, scale=100):
    """
    Plot first 9 channels of EEG interval between start and stop
    Scale parameter should be specified to visualize too big or too small amplitude of signal
    
    TODO add dynamic amount of channels
    """
    rcParams['figure.figsize'] = 15, 10
    mne.viz.plot_raw(
        raw, 
        start=start, 
        duration=stop - start, 
        show=False, 
        n_channels=9,
        scalings={'eeg': scale}
    )
    plt.show()


def plot_p300(epochs):
    """
    Plot averaged ERP signal for target and common stimuli
    """
    units = {"eeg": "mV"}
    
    epochs['1'].average().plot(units=units)
    plt.show()
    
    epochs['2'].average().plot(units=units)
    plt.show()
