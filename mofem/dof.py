import numpy
import enum
from collections import Iterable

__author__ = 'Mohamed Moussa'

def _get_index(dofset, dof):
    try: return dofset.index(dof)
    except ValueError: return None

class DOF(enum.IntEnum):
    X = 0
    Y = 1
    Z = 2
    RX = 3
    RY = 4
    RZ = 5

    def __str__(self): return self.__dict__["_name_"]
    def __repr__(self): return self.__str__()

class DOFSet(set):

    def __init__(self, *args):
        for a in args:
            self.add(a)

    def indexOf(self, dof):
        L = list(self)
        L.sort()
        if isinstance(dof, Iterable):
            return [_get_index(L, d) for d in dof]
        else:
            return _get_index(L,dof)
