from tqdm import tqdm_notebook
from sklearn.metrics import confusion_matrix
import numpy as np

from preprocess import process_signal
from models import get_dataset
from config import CHARACTER_MATRIX


def bci_score(y_true, y_pred):
    """
    Returns a value of C_cs metric for binary classification (P300 / no P300), 
    described on a page 9 of original paper
    
    According to a paper:
    'The score C_cs does not take into
    account the number of true negative examples. 
    This is important for unbalanced datasets since
    this omission helps the channel selection procedure 
    to focus on channels that give positive scores
    to positive examples which are fewer than negative examples'
    """
    matrix = confusion_matrix(y_true, y_pred)
    tp = matrix[1][1]
    fp = matrix[0][1]
    fn = matrix[1][0]
    return tp / (tp + fp + fn)


def cross_validate(sessions_df, create_model, C_values, n_classifiers, n_characters):
    """
    Performs a cross-validation for n_classifiers pipelines created with a function create_model
    and its hyperparameters placed in C_values, each one will be trained on n_characters
    
    - C_cs score is used to select the best model for the split
    - Different test sets are used to check first 7 and the rest of classifiers
    
    The cross-validation process is described on a page 10 of the original paper
    """
    hyperparameters = {}

    for classifier_index in tqdm_notebook(range(n_classifiers)):
        scores = {}

        input_indices = list(range(classifier_index * n_characters, (classifier_index + 1) * n_characters))
        train_sessions_df = sessions_df[sessions_df["InputIndex"].isin(input_indices)].reset_index(drop=True)
        train_epochs = process_signal(train_sessions_df)

        val_input_indices = [
            i for i in range(8 * n_characters)
            if i not in input_indices
        ]
        if classifier_index > 7:
            val_input_indices = [
                i for i in range(8 * n_characters, 16 * n_characters)
                if i not in input_indices
            ]

        val_sessions_df = sessions_df[sessions_df["InputIndex"].isin(val_input_indices)].reset_index(drop=True)
        val_epochs = process_signal(val_sessions_df)

        assert (len(val_epochs) == 6299) or (len(val_epochs) == 7199), len(val_epochs)

        train_X, train_y = get_dataset(train_epochs.copy())
        val_X, val_y = get_dataset(val_epochs.copy())

        for C in C_values:
            model = create_model(C)
            model.fit(train_X, train_y)
            y_pred = model.predict(val_X)

            scores[C] = (bci_score(val_y, y_pred), model)

        hyperparameters[classifier_index] = scores
        
    return hyperparameters


def select_estimators(hyperparameters):
    """
    Selects best estimator for each group of character, based on C_cs score
    """
    estimators = {}
    for index, scores in hyperparameters.items():
        max_score = 0
        best_classifier = None
        for _, (score, classifier) in scores.items():
            if score > max_score:
                max_score = score
                best_classifier = classifier
        estimators[index] = best_classifier
    return estimators


def prepare_events_starts_df(test_epochs, test_sessions_df):
    indices = test_epochs.events.T[0]
    events_start_df = test_sessions_df[["InputIndex", "Code"]].loc[indices].copy()
    events_start_df["EventIndex"] = range(events_start_df.shape[0])
    
    return events_start_df


def select_epochs_in_interval(test_epochs, events_start_df, input_index, row_column, n_samples):
    selected_indices = events_start_df[
        (events_start_df["InputIndex"] == input_index) & \
        (events_start_df["Code"] == row_column)
    ]["EventIndex"]
    
    # Removed sequences
    selected_indices = selected_indices[
        (selected_indices - selected_indices.min()) < n_samples
    ]
    
    return test_epochs[selected_indices]


def predict_characters(estimators, test_epochs, test_sessions_df, test_true_characters, n_sequences):
    """
    For each epoch given in test_epochs returns a character guessed by the first n_sequences.
    """
    n_samples = 12 * n_sequences
    starts_df = prepare_events_starts_df(test_epochs, test_sessions_df)
    predicted_characters = []

    for trial, true_character in enumerate(tqdm_notebook(test_true_characters)):
        row_predictions = []
        column_predictions = []

        for row_column in range(1, 13):
            trials_row_column = select_epochs_in_interval(test_epochs, starts_df, trial, row_column, n_samples)
            trials_X, _ = get_dataset(trials_row_column)

            trial_scores = []
            for _, estimator in estimators.items():
                score = estimator.decision_function(trials_X)
                trial_scores += [score]

            trial_prediction = np.mean(trial_scores)

            if row_column < 7:
                column_predictions.append(trial_prediction)
            else:
                row_predictions.append(trial_prediction)

        row = np.argmax(row_predictions)
        column = np.argmax(column_predictions)
        predicted_characters.append(CHARACTER_MATRIX[row][column])
    
    return predicted_characters
