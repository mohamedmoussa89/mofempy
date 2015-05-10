from numpy import inf, row_stack, array

class TemporalOutput:
    """Container to store vector values over multiple time-steps.

    Attributes
    ----------
    dt : float
        Minimum time allowed between storing of new data
    data_names : list of str
        List of names for the data stored in this object

    Notes
    -----
    The attribute list of the object is extended by what is provided in data_names.

    Examples
    --------
    >>> out = TemporalOutput(0.1, ["u", "v"])
    >>> out.add(0.0, u = array([0,0,0]), v = array([1,1,1]))
    >>> print(out.u)
    [array([0, 0, 0])]
    >>> print(out.v)
    [array([1, 1, 1])]
    """


    def __init__(self, dt, data_names):
        """Container to store vector values over multiple time-steps.

        Parameters
        ----------
        dt : float
            Minimum time allowed between storing of new time-step data
        data_names :  list of str
            List of names for the data to be stored in this object
        """

        self.dt = dt
        self.__next_t = 0

        self.data_names = data_names
        self.t = []
        for name in self.data_names:
            self.__dict__[name] = []


    def add(self, t, **data):
        """Add new vector to each of the registered names for the given timestamp.

        Attributes
        ----------
        t : float
            Timestamp for the given data.
        **data : array
            Data for each registered name to be stored

        Notes
        -----
        The vectors for each name should be the same size every time-step.
        """
        if t >= self.__next_t:

            self.t.append(t)
            for data_name, vec in data.items():
                self.__dict__[data_name].append(array(vec))
            self.__next_t += self.dt


    def finalize(self):
        """Signal end of any more data being added. Every list of scalars/arrays is converted to a single array.

        Notes
        -----
        This should be called after the simulation is complete.
        """
        self.__next_t = inf
        self.t = array(self.t)
        for name in self.data_names:
            self.__dict__[name] = row_stack(self.__dict__[name])


def test():
    dt = 0.1
    out = TemporalOutput(dt, ["u","v","a"])

    t = 0
    out.add(t, u=[], v=[], a=[])
    print(out.__dict__)

if __name__ == "__main__":
    test()



