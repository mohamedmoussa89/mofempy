__author__ = 'Mohamed Moussa'

import numpy as np

# The following is in Voigt format
I = np.array([1,1,0])

IoI = np.array([[1.0,1.0,0.0],
                [1.0,1.0,0.0],
                [0.0,0.0,0.0]])

II = np.array([[1.0,0.0,0.0],
               [0.0,1.0,0.0],
               [0.0,0.0,1.0]])

def trace(T):
    if (T.size == 6): return T[0] + T[1] + T[2]
    if (T.size == 3): return T[0] + T[1]
    raise ValueError("Tensor has incorrect number of components")