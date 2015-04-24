__author__ = 'Mohamed Moussa'

from numpy import array

def get_segments_from_nodes(nodes):
    """Pairs adjacent nodes together to form segments"""
    return zip( nodes, nodes[1:]+nodes[0:1] )


def calculate_bounding_box(nodes):
    """Calculate a bounding box"""

    x = [ n[0] for n in nodes ]
    y = [ n[1] for n in nodes ]

    return [min(x), min(y), max(x), max(y)]

def check_bounding_boxes_intersect(a, b):
    """Check if two bounding boxes intersect"""
    return a[0] <= b[2] and \
           a[2] >= b[0] and \
           a[1] <= b[3] and \
           a[3] >= b[1]


def check_segment_intersection(a,b,c,d):
    """Check if segments A-B and C-D intersect"""
    from numpy import column_stack
    from numpy.linalg import det, inv, norm

    A = column_stack([b-a, c-d])
    r = c-a

    if abs(det(A)) <= 1e-6:
        # Parallel - check bounding boxes instead
        bb_ab = calculate_bounding_box([a,b])
        bb_cd = calculate_bounding_box([c,d])
        return check_bounding_boxes_intersect(bb_ab, bb_cd)

    # Not parallel
    x = inv(A).dot(r)
    if ((0 <= x[0] <= 1) and (0 <= x[1] <= 1)):
        return True

    return False


def project_node_on_segment(node, segment):
    from numpy import dot

    a,b = segment
    c = node
    ba = b-a
    ca = c-a

    xi = dot(ca,ba) / dot(ba,ba)

    return (xi, a + (b-a)*xi)

def calculate_segment_normal(segment):
    from numpy import cross
    from numpy.linalg import norm
    p1,p2 = segment
    n = cross((p2-p1), [0,0,1.])
    return n / norm(n)

def test():
    a = array([0, 0])
    b = array([1, 0])

    c = array([1, 1])
    d = array([0, 1])

    check_segment_intersection(a,b,c,d)
    for seg_node_ids, seg_nodes  in get_edge_segments([a,b,c,d]):
        print(seg_node_ids, seg_nodes)

    a = array([0,0])
    b = array([100,0])
    c = array([1.0,0])
    xi, p = project_node_on_segment(c, [a,b])

    print(xi,p)


if __name__ == "__main__":
    test()


