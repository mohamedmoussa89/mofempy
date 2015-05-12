"""
Bucket sort data structure and search procedures

See section 3.3 of reference for details.

Reference
---------
Yastrebov, VA. (2013) Numerical Methods in Contact Mechanics
"""

from numpy import array, zeros, sqrt, empty, inf, dot, all, any
from numpy.linalg import norm

from mfpy.boundbox import calculate_bounding_box, calculate_bounding_box_intersect
from mfpy.boundbox import check_point_in_bounding_box, check_bounding_boxes_intersect

from mfpy.segment import project_node_on_segment, gather_segment_positions, calculate_segment_normals

from mfpy.contact.nodesegmentpair import NodeSegmentPair
from mfpy.contact.nodenodepair import NodeNodePair, convert_to_node_segment_pair


class BucketGrid(object):
    """
    Data structure that holds nodes and segments in a uniform bucket grid.
    Master/Slave elements are held in separate arrays
    """

    MAX_ELEMENTS_PER_CELL = 10

    def __init__(self, node_positions, segment_positions, nodes_m, nodes_s, segments_m):
        # Calculate d_max as maximum length of master segment
        self.d_max = max(norm(segment_positions[segments_m, 0:2] - segment_positions[segments_m, 2:4], axis=1))

        # Calculate spatial bounding box
        bound_box_m = calculate_bounding_box(node_positions[nodes_m], inflate= self.d_max)
        bound_box_s = calculate_bounding_box(node_positions[nodes_s], inflate= self.d_max)
        self.spatial_bb = calculate_bounding_box_intersect(bound_box_m, bound_box_s)

        # No overlap between surfaces -> Nothing to do
        if self.spatial_bb is None:
            self.master_nodes = empty(0)
            self.slave_nodes = empty(0)
            self.master_segments = empty(0)
            return

        # Cell size
        w = sqrt(2) * self.d_max

        # Split spatial bounding box into equal sized cells
        x1, y1, x2, y2 = self.spatial_bb
        delta = (x2-x1, y2-y1)
        self.cell_count = [max(int(delta[0]/w),1), max(int(delta[1]/w),1)]
        self.num_cells = self.cell_count[0] * self.cell_count[1]
        self.cell_size = [(x2-x1)/self.cell_count[0], (y2-y1)/self.cell_count[1]]

        # Array that hold distributed nodes and segments
        self.master_nodes = distribute_nodes(self, node_positions, nodes_m)
        self.slave_nodes = distribute_nodes(self, node_positions, nodes_s)
        self.master_segments = distribute_segments(self, segment_positions, segments_m)


def distribute_nodes(grid, node_positions, node_ids):
    """
    Distributes nodes into cells for a given grid.
    """
    cells = zeros((grid.num_cells, grid.MAX_ELEMENTS_PER_CELL), dtype=int)-1
    cells_count = zeros(grid.num_cells, dtype=int)
    for node_id in node_ids:
        cell_id = get_cell_number(grid, node_positions[node_id])
        if  0 <= cell_id < grid.num_cells:
            cells[cell_id, cells_count[cell_id]] = node_id
            cells_count[cell_id] += 1
    return cells


def distribute_segments(grid, segment_positions, segment_ids):
    """
    Distributes segments into cells for a given grid, based on the segments node positions.
    """
    cells = zeros((grid.num_cells, grid.MAX_ELEMENTS_PER_CELL), dtype=int)-1
    cells_count = zeros(grid.num_cells, dtype=int)
    for segment_id in segment_ids:
        node_1 = segment_positions[segment_id, 0:2]
        node_2 = segment_positions[segment_id, 2:4]
        cell_ids = set([get_cell_number(grid, node_pos) for node_pos in [node_1, node_2]])
        for cell_id in cell_ids:
            if  0 <= cell_id < grid.num_cells:
                cells[cell_id, cells_count[cell_id]] = segment_id
                cells_count[cell_id] += 1
    return cells


def get_cell_number(grid, pos):
    """
    Returns a positions corresponding cell index in a grid.
    """
    x1,y1,_,_ = grid.spatial_bb
    x,y = pos
    cell_dx, cell_dy = grid.cell_size
    return int((x-x1)/cell_dx) + int((y-y1)/cell_dy) * grid.cell_count[0]


def calculate_cell_bb(grid, cell_id):
    """
    Calculates a cells bounding box.
    """
    x1,y1,_,_ = grid.spatial_bb
    ix = cell_id % grid.cell_count[0]
    iy = int(cell_id / grid.cell_count[0])
    min_x = x1 + ix*grid.cell_size[0]
    max_x = min_x + grid.cell_size[0]
    min_y = y1 + iy*grid.cell_size[1]
    max_y = min_y + grid.cell_size[1]
    return [min_x, min_y, max_x, max_y]


def get_cell_neighbourhood(grid, cell_id):
    """
    Determines 'neighbourhood' of cells around a particular cell, including the cell itseld.
    """
    # Cells per row
    cpr = grid.cell_count[0]

    # Immediate neighbours
    neighbours = [cell_id,
                  cell_id-1,
                  cell_id+1,
                  cell_id+cpr,
                  cell_id-cpr,
                  cell_id-1+cpr,
                  cell_id-1-cpr,
                  cell_id+1+cpr,
                  cell_id+1-cpr]

    # Filter - make sure cells are valid
    neighbours = [cell_id for cell_id in neighbours if  0 <= cell_id < grid.num_cells]

    return neighbours


def find_new_contact_pairs(nsm, sem, grid, node_positions, elements, segment_positions, segment_normals):
    """
    Finds node-node or node-segment pairs that have penetrated (i.e. are in contact)
    """

    contact_pairs = []

    if grid.spatial_bb is None:
        return contact_pairs

    # For every (slave) cell
    for slave_cell_id in range(0, grid.num_cells):

        # For every slave node in this cell
        for slave_node_id in grid.slave_nodes[slave_cell_id]:
            if slave_node_id == -1: break

            # Slave position and bounding box (+-dmax around the slave node)
            slave_pos = node_positions[slave_node_id]
            slave_bb = calculate_bounding_box([slave_pos], inflate=grid.d_max)

            # Closest master found
            found = None

            segments_checked = set()

            # Cell neighbourhood
            neighbours = get_cell_neighbourhood(grid, slave_cell_id)
            for master_cell_id in neighbours:
                master_cell_bb = calculate_cell_bb(grid, master_cell_id)

                # Check intersection between slave node BB and master cell BB
                if not check_bounding_boxes_intersect(slave_bb, master_cell_bb):
                    continue

                # Slave Node-Master Segment Check ----------------------------------------------------------------------
                for master_segment_id in grid.master_segments[master_cell_id]:
                    if (master_segment_id == -1): break
                    if (master_segment_id in segments_checked): continue

                    # Get segment position and build a BB
                    segment_pos = segment_positions[master_segment_id]
                    segment_bb = calculate_bounding_box([segment_pos[0:2], segment_pos[2:4]])

                    # Check intersection of BB
                    if not check_bounding_boxes_intersect(segment_bb, slave_bb):
                        continue

                    # Get correct element and project node on to segment
                    segments_checked.add(master_segment_id)
                    elem_id, elem_segment_id = sem[master_segment_id]
                    xi, proj = elements[elem_id].project_node_on_segment(elem_segment_id, segment_pos, slave_pos)

                    # Must be within the valid range
                    if (any(xi > 1) or any(xi < -1)): continue

                    # Calculate distance to segment
                    normal = segment_normals[master_segment_id]
                    delta = slave_pos - proj
                    d = dot(delta, normal)

                    if not found or (abs(d) < grid.d_max and abs(d) < abs(found.d_min)):
                        found = NodeSegmentPair(slave_node_id, master_segment_id, normal, d, xi, proj, elem_id)
                        slave_bb = calculate_bounding_box([slave_pos], inflate=abs(d))
                #-------------------------------------------------------------------------------------------------------

                # Slave Node-Master Node Check -------------------------------------------------------------------------
                for master_node_id in grid.master_nodes[master_cell_id]:
                    if (master_node_id == -1): break

                    # First check if it is in slave BB
                    master_pos = node_positions[master_node_id]
                    if not check_point_in_bounding_box(slave_bb, master_pos):
                        continue

                    # Node is close
                    delta = slave_pos - master_pos
                    d = norm(delta)
                    if (d == 0.): continue
                    normal = -delta / d
                    # Check neighbour segments for penetration
                    penetrated = True
                    for segment_id in nsm[master_node_id]:
                        segment_normal = segment_normals[segment_id]
                        if dot(segment_normal, delta) > 0:
                            penetrated = False
                            break
                    if penetrated: d *= -1
                    if not found or (abs(d) < grid.d_max and abs(d) < abs(found.d_min)):
                        found = NodeNodePair(slave_node_id, master_node_id, normal, d)
                        # Tighten slave node BB
                        slave_bb = calculate_bounding_box([slave_pos], inflate=abs(d))
                #-------------------------------------------------------------------------------------------------------


            if found:
                contact_pairs.append(found)

    return [pair for pair in contact_pairs if pair.d_min <= 0]


def update_prev_contact_pairs(prev_contact_pairs, nsm, sem, ssm, node_positions, elements, segment_positions, segment_normals):
    """
    Check previous pairs to see if they are still in contact, or if in contact with neighbours
    """

    remaining_contact_pairs = []

    for pair in prev_contact_pairs:

        # Only look at contacting pairs
        if pair.d_min > 0: continue

        slave_pos = node_positions[pair.slave_id]

        # Determine segment neighbourhood based on pair type
        neighbours = []
        if isinstance(pair, NodeNodePair):
            neighbours = nsm[pair.master_id]
        elif isinstance(pair, NodeSegmentPair):
            neighbours = [pair.master_id] + ssm[pair.master_id]

        # Check for closest contact with neighbourhood segments
        found = None
        for parent_segment_id in neighbours:
            element_id, element_segment_id = sem[parent_segment_id]
            segment_pos = segment_positions[parent_segment_id]
            xi, proj = elements[element_id].project_node_on_segment(element_segment_id, segment_pos, slave_pos)
            if (any(xi < -1) or any(xi > 1)): continue
            normal = segment_normals[parent_segment_id]
            d = dot(slave_pos - proj, normal)
            if (d <= 0) and (abs(d) <= 2*abs(pair.d_min)+1e-12 ) and (not found or abs(d) < abs(found.d_min)):
                found = NodeSegmentPair(pair.slave_id, parent_segment_id, normal, d, xi, proj, element_id)

                # Check if we have found contact with the original segment -> done, exit immediately
                if isinstance(pair, NodeSegmentPair) and parent_segment_id == pair.master_id:
                    break

        if found:
            remaining_contact_pairs.append(found)

    return remaining_contact_pairs


def one_pass_search(nsm, snpm, sem, ssm, contact_pairs, node_positions, elements, surface_pairs, t):
    """
    One-pass search on given surface pairs

    A/B = master/slave
    """

    # Segment positions and normals
    segment_positions = gather_segment_positions(node_positions, snpm)
    segment_normals = calculate_segment_normals(segment_positions)

    # Look to see which of the previous contact pairs are still in contact
    # Slave nodes still in contact are ignored afterwards
    remaining = update_prev_contact_pairs(contact_pairs,
                                          nsm, sem, ssm,
                                          node_positions, elements,
                                          segment_positions, segment_normals)
    ignore = set([pair.slave_id for pair in remaining])

    contact_pairs = []

    for pair in surface_pairs:

        # Determine which slave nodes to search for
        slave_nodes = [slave_id for slave_id in pair.nodes_b if slave_id not in ignore]

        if slave_nodes:
            grid = BucketGrid(node_positions, segment_positions,
                              pair.nodes_a, slave_nodes,
                              pair.segments_a)

            contact_pairs.extend(find_new_contact_pairs(nsm, sem, grid,
                                                        node_positions, elements,
                                                        segment_positions, segment_normals))

    return contact_pairs + remaining