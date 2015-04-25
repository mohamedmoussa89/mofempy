from numpy import array, row_stack, inf

class KinematicOutput(object):
    """Container to store displacement, velocity and acceleration values over the course of the simulation.

    Attributes
    ----------
    t : list of float
    u : list of array
    v : list of array
    a : list of array

    """

    def __init__(self, dt):
        """Container to store displacement, velocity and acceleration values over the course of the simulation.

        Parameters
        ----------
        dt : float
            Minimum time required between timestamps
        """

        self.t = []
        self.u = []
        self.v = []
        self.a = []
        self.dt = dt

        self.__next_t = 0


    def add(self, t, u, v, a):
        """Store the displacement, velocity and acceleration vectors for the given timestamp

        Does not store the given information if the given timestamp is not larger than the minimum time between
        timestamps (dt).

        Parameters
        ----------
        t : float
            Timestamp
        u : array
            Displacement array
        v : array
            Velocity array
        a : array
            Acceleration array
        """

        if t >= self.__next_t:
            self.t.append(t)
            self.u.append(array(u))
            self.v.append(array(v))
            self.a.append(array(a))
            self.__next_t += self.dt


    def finalize(self):
        """Signal end of any more data being added. Needed to allow for efficient plotting of data.

        This should be called after the simulation is complete.
        """

        self.__next_t = inf
        self.u = row_stack(self.u)
        self.v = row_stack(self.v)
        self.a = row_stack(self.a)
