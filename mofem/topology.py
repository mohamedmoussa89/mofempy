__author__ = 'Mohamed Moussa'


def get_segments_from_nodes(nodes):
    """Pairs adjacent nodes together to form segments"""
    return zip( nodes, nodes[1:]+nodes[0:1] )


def find_boundary_segment_map(enm):
    """Find boundary segments around the model
    Maps those free segments to (Element Id, Local Segment)"""

    shared_segments = set()
    boundary_segment_map = {}

    for eid, element_enm in enumerate(enm):
        global_segments = get_segments_from_nodes(element_enm)

        local_node_ids = list(range(len(element_enm)))
        local_segments = get_segments_from_nodes(local_node_ids)

        temp = {}

        for s, local_s in zip(global_segments, local_segments):

            # Edge (or its reverse) already exists in master segments
            rev_s = tuple(reversed(s))
            if (s in boundary_segment_map) or (rev_s in boundary_segment_map):
                try: del boundary_segment_map[s]
                except: pass
                try: del boundary_segment_map[rev_s]
                except: pass
                shared_segments.add(s)

            # Edge doesnt exist yet
            # By adding to a temp, it allows for (x,y) and (y,x) to be added
            # if they are from the *same* element (eg. truss has two segments,
            # one for each side)
            elif s not in shared_segments:
                temp[s] = (eid, local_s)

        #
        boundary_segment_map.update(temp)

    return boundary_segment_map


def find_boundary_nodes(boundary_segment_map):
    return set([nid for seg in boundary_segment_map.keys() for nid in seg])
