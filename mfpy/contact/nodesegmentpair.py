"""Functionality to search for contact between nodes and segments"""

from numpy import inf, dot
from numpy.linalg import norm

from mfpy.geometry import project_node_on_segment, calculate_segment_normal

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
    elem_id : int
        Element ID of the element that owns the segment projected on
    local_seg : tuple of int
        Tuple that describes the segment based on the element local node IDs
    global_seg : tuple of int
        Tuple that describes the segment based on the global node IDs
    """

    def __init__(self, node_id, proj, xi, normal, d_min, elem_id, local_seg, global_seg):
        self.node_id = node_id
        self.proj = proj
        self.xi = xi
        self.normal = normal
        self.d_min = d_min
        self.elem_id = elem_id
        self.local_seg = local_seg
        self.global_seg = global_seg


def find_closest_segment(nodes, node_id, segment_map, only_positive=True):
    """Find closest projected point to a given node on the boundary segments given

    Parameters
    ----------
    nodes : list of array
        List of node positions#
    node_id : int
        ID of node to use for search
    segment_map : dict of tuple -> tuple
        Map of global segments to (element ID, local segment)
    only_positive : bool
        If true only search for positive distances (no penetration check). Default true.

    Returns
    -------
    found : NodeSegmentProjection
        Projection of node on the nearest segment
    """

    search_node = nodes[node_id]

    # Default projection = nothing found at infinite distance
    found = NodeSegmentPair(None, None, None, None, inf, None, None, None)

    # Check against boundary segments
    for (global_seg, (elem_id, local_seg)) in segment_map.items():
        # Dont check node against its own segments
        if (node_id in global_seg): continue

        # Get segment node positions
        seg_pos = [nodes[i] for i in global_seg]

        # Find projection on the segment
        xi, p = project_node_on_segment(search_node, seg_pos)
        if not (0 < xi < 1): continue

        # Calculate normal gap distance
        n = calculate_segment_normal(seg_pos)[0:2]
        delta = search_node - p
        d = dot(n, delta)

        # Check if we are looking only for positive values
        if (only_positive and d<0): continue

        # Update minimum results
        if (d < found.d_min):
            found = NodeSegmentPair(node_id, p, xi, n, d, elem_id, local_seg, global_seg)

    return found


def find_new_ns_pairs(ns_pairs, nodes, node_ids, segment_map, threshold):
    """Finds nodes that may collide in the future.

    Nodes that are close to a segment are considered 'active nodes'. These active nodes are
    paired with segments in a NodeSegmentPair, which is then added to an 'active list'.

    Parameters
    ----------
    active_list : list of NodeSegmentPair
        List of all NodeSegmentPairs that are currently considered active
    nodes : list of array
        List of all node positions
    node_ids : list of int
        List of all IDs of node to check if now active
    segment_map : dict of tuple -> tuple
        Map of global segments to (element ID, local segment)
    threshold : float
        Threshold distance to nearest segment to consider a node active

    Returns
    -------
    active_list : list of NodeSegmentPair
        Updated active list
    """

    # Search only in inactive nodes
    active_nodes = set([pair.node_id for pair in ns_pairs])
    inactive_nodes = set(node_ids) - active_nodes

    # Check for node-segment pairs first
    for node_id in inactive_nodes:
        pair = find_closest_segment(nodes, node_id, segment_map)

        # If close enough, add to active set
        if pair.d_min <= threshold:
            ns_pairs.append(pair)

    return ns_pairs


def update_ns_pairs(ns_pairs, nodes):
    """Updates current active node-segment pairs

    Updates the minimum distance, projection point and normal

    Parameters
    ----------
    active_list : list of NodeSegmentPair
        List of all NodeSegmentPairs that are currently considered active
    nodes : list of array
        List of all node positions

    Returns
    -------
    active_list : list of NodeSegmentPair
        Updated active list
    """

    for pair in ns_pairs:
        seg_pos = [nodes[i] for i in pair.global_seg]
        node_pos = nodes[pair.node_id]

        # Find projection on each master segment
        pair.xi, pair.proj = project_node_on_segment(node_pos, seg_pos)

        # Calculate normal gap distance
        pair.normal = calculate_segment_normal(seg_pos)[0:2]
        delta = node_pos - pair.proj
        pair.d_min = dot(pair.normal, delta)

    return ns_pairs


def remove_ns_pairs(ns_pairs, threshold):
    """Makes nodes inactive if they move away from the segment based on given threshold

    An active pair is made inactive if the distance has increased beyond the threshold or the projection
    of the node is no longer on the segment.

    Parameters
    ----------
    active_list : list of NodeSegmentPair
        List of all NodeSegmentPairs that are currently considered active
    threshold : float
        Threshold distance to nearest segment to consider a node active

    Returns
    -------
    active_list : list of NodeSegmentPair
        Updated active list
    """
    return [pair for pair in ns_pairs if (pair.d_min < threshold) and (0 < pair.xi < 1)]


def get_penetrating_active_pairs(active_list):
    """ Returns only the NodeSegmentPairs where the node is penetrating the segment"""
    return [ap for ap in active_list if ap.d_min <= 0]