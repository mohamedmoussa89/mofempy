from numpy import zeros, dot, inf
from numpy.linalg import norm


class NodeNodePair(object):
    def __init__(self, node_id, other_id, normal, d_min):
        self.node_id = node_id
        self.other_id = other_id
        self.normal = normal
        self.d_min = d_min


def find_closest_node(nodes, search_node_id, node_ids):
    """Find closest boundary node to a given  node

    Parameters
    ----------
    nodes : list of array
        List of node positions
    search_node_id : int
        ID of node to use for search
    node_ids : list of int
        List of node IDs that will be searched against

    Returns
    ------
    found : NodeNodePair
        Node-node pair
    """

    search_node = nodes[search_node_id]

    found = NodeNodePair(search_node_id, None, None, inf)

    # Check against all other nodes
    for id in node_ids:
        if (id == search_node_id): continue

        other_node = nodes[id]

        delta = other_node-search_node
        normal = delta / norm(delta)
        d = norm(delta)
        if (d < found.d_min):
            found.d_min = d
            found.other_id = id
            found.normal = normal

    return found


def find_new_nn_pairs(nn_pairs, nodes, node_ids, threshold):

    # Only search inactive nodes
    active_nodes = set([pair.node_id for pair in nn_pairs])
    inactive_nodes = set(node_ids) - active_nodes

    for node_id in inactive_nodes:
        pair = find_closest_node(nodes, node_id, node_ids)

        if pair.d_min <= threshold:
            nn_pairs.append(pair)

    return nn_pairs


def update_nn_pairs(nn_pairs, nodes):
    for pair in nn_pairs:
        delta = nodes[pair.node_id] - nodes[pair.other_id]
        pair.d_min = norm(delta)
        pair.normal = delta/norm(delta)

    return nn_pairs

def remove_nn_pairs(nn_pairs, threshold):
    return [pair for pair in nn_pairs if pair.d_min < threshold]