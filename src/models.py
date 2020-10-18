from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import json


def get_dataset(epochs):
    """
    Prepare dataset for the classification using signal features
    - Decimate epochs
    - Reshape to use the signal in sklearn Estimator
    
    The decimation factor = 12, because, according to a paper, each epoch was band-pass filtered to 0.1-10Hz, 
    and the minimum sample rate that will represent such signal without an aliasing is 20Hz,
    so we need to reduce the original sampling frequency of 240Hz in 12 times
    """
    epochs = epochs.decimate(12)
    
    events_X = epochs.get_data()
    events_X = events_X.reshape(events_X.shape[0], -1)
    
    events_y = epochs.events.T[-1]
    
    return np.vstack(events_X), (events_y == 1).astype(int)


def create_svc_pipeline(C):
    """
    Creates a pipeline for each SVM classifier
    """
    pipe = Pipeline([
        ('scaler', StandardScaler()), 
        ('svm', SVC(gamma='auto', kernel="linear", cache_size=2000, C=C))
    ])
    return pipe
