from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import joblib

import json

import numpy as np

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV


# # TODO should be refactored

def get_dataset(epochs):
    """
    Prepare dataset for the classification using signal features
    - Decimate epochs
    - Reshape to use the signal in sklearn Estimator
    """
    epochs = epochs.decimate(12)
    
    events_X = epochs.get_data()
    events_X = events_X.reshape(events_X.shape[0], -1)
    
    events_y = epochs.events.T[-1]
    
    return np.vstack(events_X), (events_y == 1).astype(int)


class P300Classifier():
    model = None
    grid = None
    search = None
    packages = None
    channels = None
    sfreq = None
    
    def __init__(self):
        self.search = GridSearchCV(
            self.model, 
            self.grid,
            scoring="roc_auc",
            cv=5
        )
        
    def fit(self, epochs, copy=True):
        if copy:
            epochs = epochs.copy()
        self.packages = epochs.get_data().shape[-1]
        self.channels = epochs.ch_names
        self.sfreq = epochs.info["sfreq"]
        X, y = get_dataset(epochs)
        
        self.model.fit(X, y)
    
    def predict(self, epochs, copy=True):
        if copy:
            epochs = epochs.copy()
        
        assert epochs.get_data().shape[-1] == self.packages
        assert all([a == b for a, b in zip(epochs.ch_names, self.channels)])
        assert epochs.info["sfreq"] == self.sfreq
        X, y = get_dataset(epochs)
        return self.model.predict(X), y
    
    def save(self, directory):
        configuration = {
            "channels": self.channels,
            "packages": self.packages,
            "sfreq": self.sfreq
        }
        json.dump(configuration, open(directory + "/conf.json", "w"))
        joblib.dump(self.model, directory + "/model.pkl")
    
    def load(self, directory):
        configuration = json.load(open(directory + "/conf.json"))
        self.channels = configuration["channels"]
        self.packages = configuration["packages"]
        self.sfreq = configuration["sfreq"]
        self.model = joblib.load(directory + "/model.pkl")


class P300ClassifierSVM(P300Classifier):
    def __init__(self):
        self.model = SVC()
        self.grid = {}
        super().__init__()


class P300ClassifierLDA(P300Classifier):
    def __init__(self):
        self.model = LDA()
        self.grid = {
            'C': [0.1]
        }
        super().__init__()


def get_scores(events_X, events_y, window_size=15):
    """
    TODO should be refactored
    
    Get roc auc scores for LDA model trained on all events except events in the certain window
    Perform this prediction in a sliding window manner
    """
    all_indices = np.arange(events_y.shape[0])
    time = np.linspace(-100, 700, events_X.shape[-1] // 8)
    
    scores = []
    
    positive_events_indices = np.nonzero(events_y == 1)[0]

    for i in range(0, len(positive_events_indices), window_size):
        positive_window = positive_events_indices[i:i + window_size]
        window_start = positive_window.min()
        window_end = positive_window.max()
        chosen_indices = (all_indices < window_start) | (all_indices > window_end)
        included_events_X = events_X[chosen_indices]
        included_events_y = events_y[chosen_indices]

        model = LDA()
        model.fit(included_events_X, included_events_y)

        excluded_y = events_y[~chosen_indices]
        excluded_X = events_X[~chosen_indices]
        excluded_predictions = model.predict_proba(excluded_X).T[1]
        roc_auc = roc_auc_score(excluded_y, excluded_predictions)
        scores.append(roc_auc)
    return scores
