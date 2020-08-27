from scipy.io import loadmat
import numpy as np
import pandas as pd

from config import *


def convert_file(subject):
    """
    Legacy, for g.tec dataset only
    """
    session = loadmat('{}/{}.mat'.format(DATA_PATH, subject))

    channels = session[subject]['train'][0][0][1:9]

    trigger = (session[subject]['train'][0][0][9] > 0).astype(int)
    trigger[(trigger > 0) & (session[subject]['train'][0][0][10] == 0)] -= 2

    converted_session = {
        '__globals__': [],
         '__header__': b'MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Thu Jun 25 11:26:40 2020',
         '__version__': '1.0',
        'fs': np.array([[FREQUENCY]], dtype=np.uint8),
        'trig': trigger[:, np.newaxis],
        'y': channels.T
    }

    return converted_session


def convert_bci_iii_file(subject):
    """
    Load sessions information for Subject_{Subject}_{Dataset}
    Where:
    - Subject - A or B
    - Dataset - Train or Test
    
    Returns:
    - Matlab dictionary converted to a readable format
    - Set of target characters
    """
    session = loadmat('{}/{}.mat'.format(DATA_PATH, subject))

    channels = np.vstack(session['Signal']).swapaxes(0, 1)
    
    trigger = np.hstack(session["StimulusCode"])
    
    # TODO check if it is nessesary to store information about StimulusType
#     stimulus_type = np.hstack(session['StimulusType'])
    
#     trigger[trigger == -0] = 0
#     trigger[(trigger > 0) & (stimulus_type == 0)] -= 2
    # TODO Read channels from a separate file
    
    inputs, times, _ = session["Signal"].shape
    input_index = np.hstack(np.array([range(0, inputs)] * times).T)
    
    converted_session = {
        '__globals__': [],
         '__header__': b'MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Thu Jun 25 11:26:40 2020',
         '__version__': '1.0',
        'fs': np.array([[FREQUENCY]], dtype=np.uint8),
        'trig': trigger[:, np.newaxis],
        'y': channels.T,
        'input': input_index
    }

    return converted_session, session['TargetChar']


def create_df(session):
    """
    Load given dictionary with channels info and dictionary into the dataframe
    For field Trigger, will contain 1 for target stimulus and 2 for other stimulus
    """
    channels = CHANNELS
    
    session_df = pd.DataFrame(session["y"], columns=channels)
    session_df["Time"] = session_df.index / session["fs"][0][0]
    session_df["Trigger"] = session["trig"]
    if 'input' in session:
        session_df["InputIndex"] = session['input']
    session_df.loc[session_df["Trigger"] == -1, "Trigger"] = 2
    
    return session_df


def load_data(i):
    """
    Legacy, for g.tec dataset only 
    """
    subject = "s{}".format(i)
    channels = CHANNELS
    
    session = convert_file(subject)
    session_df = create_df(session)
    
    return session_df


def load_bci_iii_data(subject):
    """
    Loads BCI III dataset for a given subject into dataframe.
    Returns target characters for each attemp as well
    """
    channels = CHANNELS
    
    session, characters = convert_bci_iii_file(subject)
    session_df = create_df(session)
    
    return session_df, characters
