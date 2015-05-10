"""
A segment is defined by two nodes
"""

from numpy import empty, hstack, zeros, cross, dot
from numpy.linalg import norm


def calculate_npsm(node_pairs):
    """
    Node Pair to Segment map

    Maps directed segments (node pairs) to a unique ID
    """

    pair_to_segment = {}
    segment_id = 0
    node_pairs = set(node_pairs)
    for pair in node_pairs:
        pair_to_segment[pair] = segment_id
        segment_id += 1

    return pair_to_segment


def calculate_snpm(npsm):
    """
    Segment to Node Pair map

    Maps segment ID to directed segments (node pairs)
    """

    snpm = empty( (len(npsm), 2) , dtype=int)

    for pair, segment_id in npsm.items():
        snpm[segment_id] = pair

    return snpm


def calculate_nsm(snpm, num_nodes):
    """
    Node Segment Map

    Maps a node ID to the IDs of the segments it is connected to.
    """

    # Every node can have two connected segments in 2D
    nsm = zeros( (num_nodes, 2), dtype=int) - 1

    for i, node_pair in enumerate(snpm):

        for node_id in node_pair:
            if nsm[node_id][0] == -1:
                nsm[node_id][0] = i

            elif nsm[node_id][1] == -1:
                nsm[node_id][1] = i

            else:
                raise ValueError("Too many segments attached to node %d" % node_id)

    return nsm


def calculate_sem(npsm, elements, num_segments):
    """
    Segment Element Map

    Maps a segment to the element and its (local) element segment number
    S_ID -> [E_ID, ES_ID]
    """
    sem = zeros( (num_segments, 2), dtype=int ) - 1

    for node_pair, global_segment_id in npsm.items():
        for element_id, element in enumerate(elements):
            element_segment_id = element.get_segment_local_id(node_pair)
            if element_segment_id is not None:
                sem[global_segment_id] = [element_id, element_segment_id]
                break

    return sem


def calculate_ssm(nsm, snpm):
    """
    Segment Segment Map

    Maps a segment to the segments neighbourhood
    """
    ssm = []
    for segment_id, segment_nodes in enumerate(snpm):
        neighbours = []
        for child_node_id in segment_nodes:
            for parent_segment_id in nsm[child_node_id]:
                if parent_segment_id == -1: continue
                if parent_segment_id == segment_id: continue
                if parent_segment_id in neighbours: continue
                neighbours.append(parent_segment_id)
        ssm.append(neighbours)
    return ssm


def gather_segment_positions(node_positions, snpm):
    """
    Gather segment positions into one array
    """
    num_segments = snpm.shape[0]

    # Two nodes in two dimensions = 4
    segment_positions = empty( (num_segments,  4) )

    for segment_id in range(0, num_segments):
        segment_nodes = snpm[segment_id]
        segment_positions[segment_id, :] = hstack( [node_positions[node_id] for node_id in segment_nodes] )

    return segment_positions


def calculate_segment_normal(position):
    """
    Given a segment position, calculates its normal

    Assumes directed segment pointing from the first node to the second node
    """
    n = cross((position[2:4]-position[0:2]), [0,0,1.])
    return n / norm(n)


def calculate_segment_normals(segment_positions):
    """
    Calculates all segment normals
    """

    num_segments = segment_positions.shape[0]
    segment_normals = empty( (num_segments, 2) )

    for i, pos in enumerate(segment_positions):
        segment_normals[i] = calculate_segment_normal(pos)[0:2]

    return segment_normals


def calculate_segment_lengths(segment_positions):
    """
    Calculates all segment lengths
    """
    # Subtract nodes from each other, then find norm along the second axis
    return norm(segment_positions[:,2:4] - segment_positions[:,0:2], axis=1)


def project_node_on_segment(node, position):
    """
    Projects a node on to a (linear) segment
    """
    a,b = position[0:2], position[2:4]
    c = node
    ba = b-a
    ca = c-a

    xi = dot(ca,ba) / dot(ba,ba)

    return (xi, a + (b-a)*xi)