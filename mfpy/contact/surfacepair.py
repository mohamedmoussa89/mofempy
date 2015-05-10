class SurfacePair(object):
    """
    Data structure describing a pair of surfaces that may contact each other.

    The master/slave terms are not used here, as this could be used in a two-pass contact algorithm.
    Instead, the surfaces are named A/B. How they are handled depends on the overall contact algorithm.

    Attributes
    ----------
    nodes_a : array of int
        Node IDs for surface 'A'
    nodes_b : array of int
        Node IDs for surface 'B'

    segments_a : array of int
        Segment IDs for surface 'A'
    segments_b : array of int
        Segment IDs for surface 'B'
    """

    def __init__(self, nodes_a, nodes_b, segments_a, segments_b):
        # Node IDs for each surface
        self.nodes_a = nodes_a
        self.nodes_b = nodes_b

        # Segments IDs for each surface
        self.segments_a = segments_a
        self.segments_b = segments_b

    def __str__(self):
        return "SURFACE_PAIR(NODES %s/%s : SEGS %s/%s)" % (self.nodes_a, self.nodes_b, self.segments_a, self.segments_b)

    def __repr__(self):
        return self.__str__()


def create_surface_pairs(npsm, surfaces, pairs):
    """
    Calculates SurfacePairs from surfaces and pair indices provided

    Attributes
    ----------
    surface_nodes :
        Lists of surfaces. For example, the following defines two surfaces (with 2 and 3 segments) --
        [
         [(0,1), (1,2)],
         [(3,4), (4,5), (5,6)]
        ]

    pairs :
        List of index tuples that specify which surface to pair up.
        For example, the following defines two pairs of surfaces. The indices refer to surface_nodes --
        [(0,1), (2,3)]
    """

    surface_pairs = []

    for a,b in pairs:

        nodes_a = list(set([node_id for segment in surfaces[a] for node_id in segment]))
        nodes_b = list(set([node_id for segment in surfaces[b] for node_id in segment]))

        segments_a = [npsm[pair] for pair in surfaces[a]]
        segments_b = [npsm[pair] for pair in surfaces[b]]

        surface_pairs.append(SurfacePair(nodes_a, nodes_b, segments_a, segments_b))

    return surface_pairs
