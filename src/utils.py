import numpy as np

from config import TIME as time


def convert_time_to_sample(x):
    return np.argmin(np.abs(time - x))


def convert_sample_to_time(i):
    return time[i]
