import numpy as np

from config import TIME as time, CHARACTER_MATRIX


def convert_time_to_sample(x):
    return np.argmin(np.abs(time - x))


def convert_sample_to_time(i):
    return time[i]


def get_row_column_for_character(c):
    index = "".join(CHARACTER_MATRIX).index(c)
    return index % 6 + 1, index // 6 + 7