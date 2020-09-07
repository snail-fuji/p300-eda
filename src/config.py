import numpy as np

# +
# Configuration for g.tec dataset
# FREQUENCY = 250
# CHANNELS = ["Fz", "Cz", "P3", "Pz", "P4", "Po7", "Po8", "Oz"]
# DATA_PATH = "../data"
# -

# Configuration for BCI III dataset

CHARACTER_MATRIX = """
ABCDEF
GHIJKL
MNOPQR
STUVWX
YZ1234
56789_
""".strip().split("\n")

FREQUENCY = 240
CHANNELS = [
    "Fc5", "Fc3", "Fc1", "Fcz", "Fc2", "Fc4", "Fc6", 
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6", 
    "Cp5", "Cp3", "Cp1", "Cpz", "Cp2", "Cp4", "Cp6", 
    "Fp1", "Fpz", "Fp2", 
    "Af7", "Af3", "Afz", "Af4", "Af8", 
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8", 
    "Ft7", "Ft8", 
    "T7", "T8", "T9", "T10", 
    "Tp7", "Tp8", 
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", 
    "Po7", "Po3", "Poz", "Po4", "Po8", 
    "O1", "Oz", "O2", 
    "Iz"
]
DATA_PATH = "../data/BCI_Comp_III_Wads_2004"

START_TIME = -100
END_TIME = 700
N_SAMPLES = int((END_TIME - START_TIME) / 1000 * FREQUENCY)
TIME = np.linspace(-START_TIME, END_TIME, N_SAMPLES + 1)
