class NodeSegmentPair(object):
    """Container that describes a node/segment pair and the projection of the node on to the segment.

    Attributes
    ----------
    node_id : int
        Node ID
    proj : array
        Node where projection is
    xi : float
        Local coordinate of the projection on the segment
    normal : array
        Normal vector of the segment at the point of projection
    global_seg : tuple of int
        Tuple that describes the segment based on the global node IDs
    """

    def __init__(self, slave_id, master_id, normal, d_min, xi, proj, element_id):
        self.slave_id = slave_id
        self.master_id = master_id
        self.normal = normal
        self.d_min = d_min
        self.proj = proj
        self.xi = xi
        self.element_id = element_id

    def __str__(self):
        return "NODE_SEGMENT_PAIR(SLAVE %d, MASTER %d, NORMAL %s, D_MIN %.3f, XI %s, PROJ %s, ELEM %d)" % \
               (self.slave_id, self.master_id, self.normal, self.d_min, self.xi, self.proj, self.element_id)
    def __repr__(self): return self.__str__()