# # TODO should be refactored

def get_dataset(i):
    """
    Prepare dataset for the classification using signal features
    - Decimate epochs
    - Reshape to use the signal in sklearn Estimator
    """
    epochs = process_signal(i)
    epochs = epochs.decimate(12)
    
    events_X = epochs.get_data()
    events_X = events_X.reshape(events_X.shape[0], -1)
    
    events_y = epochs.events.T[-1]
    
    return np.vstack(events_X), (events_y == 1).astype(int)


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
