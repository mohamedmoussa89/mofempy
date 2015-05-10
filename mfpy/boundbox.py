def calculate_bounding_box(nodes, inflate = 0):
    """
    Calculate a bounding box
    """

    x = [ n[0] for n in nodes ]
    y = [ n[1] for n in nodes ]

    return [min(x)-inflate, min(y)-inflate, max(x)+inflate, max(y)+inflate]


def calculate_bounding_box_intersect(a, b):
    """
    Return intersection of two bounding boxes, if it exists
    """
    xL = max( a[0], b[0] )
    yB = max( a[1], b[1] )
    xR = min( a[2], b[2] )
    yT = min( a[3], b[3] )
    if xL < xR and yB < yT:
        return [xL, yB, xR, yT]
    else:
        return None


def check_bounding_boxes_intersect(a, b):
    """
    Check if two bounding boxes intersect
    """
    xL = max( a[0], b[0] )
    yB = max( a[1], b[1] )
    xR = min( a[2], b[2] )
    yT = min( a[3], b[3] )
    return (xL <= xR) and (yB <= yT)


def check_point_in_bounding_box(bb, p):
    """
    Check if a point is in a bounding box
    """
    min_x, min_y, max_x, max_y = bb
    x,y = p
    return (min_x <= x <= max_x and min_y <= y <= max_y)