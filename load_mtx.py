import numpy as np


def load_mtx(filename):
    with open(filename) as f:
        lines = f.readlines()
        return np.array([np.fromstring(i, sep=' ', dtype=np.float64) for i in lines[:-1]]), np.fromstring(lines[-1], sep=' ', dtype=np.float64)