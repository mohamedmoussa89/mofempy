class NodeNodePair(object):
    """Container that describes a node/node pair

    Attributes
    ----------
    slave_id : int
        Slave node ID
    master_id : int
        Master node ID
    normal : array
        Normal vector
    d_min  : float
        Penetration (negative)
    """

    def __init__(self, slave, master, normal, d):
        self.slave_id = slave
        self.master_id = master
        self.normal = normal
        self.d_min = d

    def __str__(self):
        return "NODE_NODE_PAIR(SLAVE %d, MASTER %d, NORMAL %s, D_MIN %s)" % (self.slave_id, self.master_id,
                                                                             self.normal, self.d_min)
    def __repr__(self): return self.__str__()


from .nodesegmentpair import NodeSegmentPair
def convert_to_node_segment_pair(nsm, snpm, pair):
    """
    Converts a NodeNodePair to a NodeSegmentPair
    """

    if isinstance(pair, NodeSegmentPair): return pair

    # Get first parent segment for this node
    master_segment_id = nsm[pair.master_id][0]
    if (master_segment_id == -1):
        raise ValueError("Node %s does not have a parent segment" % pair.slave_id)

    # Determine correct xi value
    xi = 0.0
    if snpm[master_segment_id][1] == pair.master_id:
        xi = 1.0

    return NodeSegmentPair(pair.slave_id, master_segment_id, pair.proj, xi, pair.normal, pair.d_min)