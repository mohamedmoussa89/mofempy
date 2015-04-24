__author__ = 'Mohamed Moussa'

import numpy as np

global_cs_2d = np.identity(2)
global_cs_3d = np.identity(3)

def create(a, b=None, c=None):
    """Create a coordinate system from given vectors"""
    a /= np.linalg.norm(a)
    if not b: return a[np.newaxis].transpose()

    b /= np.linalg.norm(b)
    if not c: return np.vstack((a,b)).transpose()

    c /= np.linalg.norm(c)
    return np.vstack((a,b,c)).transpose()


def calc_transform(A,B):
    """Calculate transformation matrix from system A to system B"""
    dimA = A.shape[1]
    dimB = B.shape[1]

    # Calculate the transform matrix
    T = np.zeros([dimB,dimA])
    for j in range(0,dimB):
        for i in range(0,dimA):
            T[j,i] = np.dot(B[:,j], A[:,i])

    return T.squeeze()

if __name__ == "__main__":

    local = create([1,1])
    T = calc_transform(global_cs_2d, local)

    print(T)
    u = T.dot([1,1])
    print(u)