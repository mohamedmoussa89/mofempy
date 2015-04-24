__author__ = 'Mohamed Moussa'

from numpy import array

class TemporalData(list):
    def __init__(self, dt):
        self.t = 0
        self.dt = dt

    def append(self, t, val):
        if (t >= self.t):
            super().append((t,val))
            self.t += self.dt

class KinematicOutput:
    def __init__(self, dt):
        self.t = []
        self.u = []
        self.v = []
        self.a = []

        self.dt = dt
        self.next_t = 0

    def add(self, t, u, v, a):
        if t >= self.next_t:
            self.t.append(t)
            self.u.append(array(u))
            self.v.append(array(v))
            self.a.append(array(a))
            self.next_t += self.dt

    def finalize(self):
        from numpy import row_stack, array
        self.u = row_stack(self.u)
        self.v = row_stack(self.v)
        self.a = row_stack(self.a)
