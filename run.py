import argparse
import time
import numpy as np
import mne

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations

import sys
sys.path.append("src")

from preprocess import process_raw_signal
from models import P300ClassifierLDA

sfreq = BoardShim.get_sampling_rate (BoardIds.CYTON_BOARD.value)
eeg_channels = BoardShim.get_eeg_channels (BoardIds.CYTON_BOARD.value)


MICROSECONDS_BEFORE_STIMULUS = 100
MICROSECONDS_STIMULUS = 10
MICROSECONDS_AFTER_STIMULUS = 700 - MICROSECONDS_STIMULUS
MICROSECONDS_TOTAL = MICROSECONDS_BEFORE_STIMULUS + MICROSECONDS_STIMULUS + MICROSECONDS_AFTER_STIMULUS
MICROSECONDS_FILTER_CALIBRATION = 7000
MICROSECONDS_EPS = 5
SAMPLES_TOTAL = (MICROSECONDS_FILTER_CALIBRATION + MICROSECONDS_TOTAL) * sfreq // 1000


def show_stimulus():
    print("Stimulus", end="\r")
    time.sleep(MICROSECONDS_STIMULUS / 1000)
    print("        ", end="\r")


def create_raw(data, model):
    eeg_data = data[eeg_channels, :]
    eeg_data = eeg_data / 1000000
    # eeg_data = eeg_data / 1000000 # BrainFlow returns uV, convert to V for MNE

    # Creating MNE objects from brainflow data arrays
    trigger_data = eeg_data[0] * 0
    trigger_data[- sfreq * (MICROSECONDS_AFTER_STIMULUS + MICROSECONDS_STIMULUS + MICROSECONDS_EPS) // 1000] = 1
    # TODO concat eeg_data and trigger_data

    eeg_data = np.vstack([[trigger_data], eeg_data])

    ch_names = ['Trigger'] + model.channels # BoardShim.get_eeg_names (BoardIds.CYTON_BOARD.value)
    ch_types = ['stim'] + ['eeg'] * len (eeg_channels)
    info = mne.create_info (ch_names = ch_names, sfreq = sfreq, ch_types = ch_types)

    raw = mne.io.RawArray (eeg_data, info)

    return raw


def prepare_raw(raw, model):
    raw.resample(model.sfreq)
    epoch = process_raw_signal(raw)
    return model.predict(epoch)


def main():
    parser = argparse.ArgumentParser ()
    # use docs to check which parameters are required for specific board, e.g. for Cyton - set serial port
    parser.add_argument ('--timeout', type = int, help  = 'timeout for device discovery or connection', required = False, default = 0)
    parser.add_argument ('--ip-port', type = int, help  = 'ip port', required = False, default = 0)
    parser.add_argument ('--ip-protocol', type = int, help  = 'ip protocol, check IpProtocolType enum', required = False, default = 0)
    parser.add_argument ('--ip-address', type = str, help  = 'ip address', required = False, default = '')
    parser.add_argument ('--serial-port', type = str, help  = 'serial port', required = False, default = '')
    parser.add_argument ('--mac-address', type = str, help  = 'mac address', required = False, default = '')
    parser.add_argument ('--other-info', type = str, help  = 'other info', required = False, default = '')
    parser.add_argument ('--streamer-params', type = str, help  = 'streamer params', required = False, default = '')
    parser.add_argument ('--serial-number', type = str, help  = 'serial number', required = False, default = '')
    parser.add_argument ('--board-id', type = int, help  = 'board id, check docs to get a list of supported boards', required = True)
    parser.add_argument ('--log', action = 'store_true')
    args = parser.parse_args ()

    params = BrainFlowInputParams ()
    params.ip_port = args.ip_port
    params.serial_port = args.serial_port
    params.mac_address = args.mac_address
    params.other_info = args.other_info
    params.serial_number = args.serial_number
    params.ip_address = args.ip_address
    params.ip_protocol = args.ip_protocol
    params.timeout = args.timeout

    if (args.log):
        BoardShim.enable_dev_board_logger ()
    else:
        BoardShim.disable_board_logger ()

    board = BoardShim (args.board_id, params)
    board.prepare_session ()

    board.start_stream (45000, args.streamer_params)

    model = P300ClassifierLDA()
    model.load("test-model")

    stopped = True
    raw = None

    time.sleep(3)

    while stopped:
        try:
            # time.sleep(MICROSECONDS_BEFORE_STIMULUS / 1000)
            # show_stimulus()
            # time.sleep(MICROSECONDS_AFTER_STIMULUS / 1000)
            data = board.get_current_board_data(SAMPLES_TOTAL) # TODO constant from model
            raw = create_raw(data, model)
            prediction = prepare_raw(raw, model)        
            print("Prediction: {}".format(prediction))
        except KeyboardInterrupt:
            print("Got keyboard interrupt, stopping...")
            break

    board.stop_stream ()
    board.release_session ()


if __name__ == "__main__":
    main()
