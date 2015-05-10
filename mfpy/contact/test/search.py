import pyglet as pyg

def test():
    from numpy import array, float64, linspace

    #x = [0,1,2,3,4, 5,6,7, 8, 9,10,     0]
    #y = [0,0,0,1,0,-1,0,0,-1,-1, 0,   0.1]
    x = [-1,5,4,0,     0]
    y = [0,0,1,1,   0.1]
    nodes = [array([n_x, n_y], dtype=float64) for n_x, n_y in zip(x,y)]

    node_ids = list(range(len(nodes)-1))
    #segments = [ (n1,n2) for n1,n2 in zip(node_ids[:-1],node_ids[1:]) ]
    #segments.extend( [ (n1,n2)for n1,n2 in zip(node_ids[:-1],node_ids[1:]) ])
    segments = [(0,1),(1,2),(2,3),(3,0)]

    node_ids = list(range(len(nodes)))
    nsm = calculate_nsm(segments)

    global external, internal, active, last_normal, i
    i = 0
    external = None
    internal = None
    threshold = 0.9
    active = False
    last_normal = None

    global neigh_seg
    neigh_seg = []

    def update_pos(x,y):
        global external, internal, neigh_seg, active
        nodes[4] = array([x,y], dtype = float64)

        #d_min = 0
        #if active and internal: d_min = 1.5*abs(direct_int.d_min)

        direct_ext, direct_int = find_segments_directly(nodes, segments, 4, threshold)
        indirect_ext, indirect_int = find_segments_indirectly(nodes, node_ids, nsm, 4, threshold)

        # Direct internal and external map to the same segment
        # This means we are directly on a segment, and outside/inside is not defined
        if (direct_ext and direct_int and direct_ext.global_seg == direct_int.global_seg):
            # Check if in neighbourhood
            active = True
            internal = direct_int
            (a,b) = direct_ext.global_seg
            if (a,b) in neigh_seg:
                neigh_seg = get_segment_neighbourhood((a,b), nsm)
            elif (b,a) in neigh_seg:
                neigh_seg = get_segment_neighbourhood((b,a), nsm)
            else: print("UNKNOWN")
            return

        external = direct_ext
        if not direct_ext: external = indirect_ext
        elif indirect_ext and abs(direct_ext.d_min) > abs(indirect_ext.d_min): external = indirect_ext

        internal = direct_int
        if not direct_int: internal = indirect_int
        elif indirect_int and abs(direct_int.d_min) > abs(indirect_int.d_min): internal = indirect_int

        if not active:
            if internal and (internal.global_seg in neigh_seg or not external):
                active = True
                neigh_seg = get_segment_neighbourhood(internal.global_seg, nsm)

            elif external:
                neigh_seg = get_segment_neighbourhood(external.global_seg, nsm)

            else:
                neigh_seg = []

        elif active:
            if external and (external.global_seg in neigh_seg or not internal):
                active = False
                neigh_seg = get_segment_neighbourhood(external.global_seg, nsm)

            elif internal:
                neigh_seg = get_segment_neighbourhood(internal.global_seg, nsm)

            else:
                active = False
                neigh_seg = []



    # Super hacky rendering
    window = pyg.window.Window()
    width = window.width
    height = window.height
    scale = (0.15, 0.15)
    trans = (-5., 0.)

    def win_to_world(x,y):
        h = height
        w = width
        sx, sy = scale
        tx, ty = trans
        return ((2/w*x-1)/sx - tx, (2/h*y-1)/sy - ty)

    def world_to_win(x,y):
        h = height
        w = width
        sx, sy = scale
        tx, ty = trans
        return ( w/2*((x+tx)*sx +1), h/2*((y+ty)*sy +1)  )

    @window.event
    def on_draw():
        pyg.gl.glClear(pyg.gl.GL_COLOR_BUFFER_BIT)

        pyg.gl.glMatrixMode(pyg.gl.GL_PROJECTION)
        pyg.gl.glLoadIdentity()
        pyg.gl.glScalef(scale[0], scale[1], 1.)
        pyg.gl.glTranslatef(trans[0], trans[1], 0.)
        pyg.gl.glPointSize(5.0)

        pyg.gl.glColor3f(1,1,1)
        pyg.gl.glBegin(pyg.gl.GL_POINTS)
        for n in nodes:
            pyg.gl.glVertex2f(n[0], n[1])
        pyg.gl.glEnd()

        pyg.gl.glBegin(pyg.gl.GL_LINES)
        for s in segments:
            n1 = nodes[s[0]]
            n2 = nodes[s[1]]
            pyg.gl.glVertex2f(n1[0], n1[1])
            pyg.gl.glVertex2f(n2[0], n2[1])
        pyg.gl.glEnd()

        global neigh_seg
        for seg in neigh_seg:
            seg_pos = [nodes[i] for i in seg]
            pyg.gl.glColor3f(1,1,0)
            pyg.gl.glBegin(pyg.gl.GL_LINES)
            pyg.gl.glVertex2f(seg_pos[0][0], seg_pos[0][1])
            pyg.gl.glVertex2f(seg_pos[1][0], seg_pos[1][1])
            pyg.gl.glEnd()
            n = calculate_segment_normal(seg_pos)
            mid = sum(seg_pos)/2
            pyg.gl.glBegin(pyg.gl.GL_LINES)
            pyg.gl.glVertex2f(mid[0], mid[1])
            pyg.gl.glVertex2f(mid[0]+n[0], mid[1]+n[1])
            pyg.gl.glEnd()

        global external
        if not active and external:
            seg_pos = [nodes[i] for i in external.global_seg]
            pyg.gl.glColor3f(0,1,0)
            pyg.gl.glBegin(pyg.gl.GL_LINES)
            pyg.gl.glVertex2f(seg_pos[0][0], seg_pos[0][1])
            pyg.gl.glVertex2f(seg_pos[1][0], seg_pos[1][1])
            pyg.gl.glEnd()

        global internal
        if active and internal:
            seg_pos = [nodes[i] for i in internal.global_seg]
            pyg.gl.glColor3f(1,0,0)
            pyg.gl.glBegin(pyg.gl.GL_LINES)
            pyg.gl.glVertex2f(seg_pos[0][0], seg_pos[0][1])
            pyg.gl.glVertex2f(seg_pos[1][0], seg_pos[1][1])
            pyg.gl.glEnd()
            x,y = nodes[internal.node_id]
            n = internal.normal
            pyg.gl.glBegin(pyg.gl.GL_LINES)
            pyg.gl.glVertex2f(x, y)
            pyg.gl.glVertex2f(x+n[0], y+n[1])
            pyg.gl.glEnd()


    @window.event
    def on_mouse_motion(x, y, button, modifiers):
        x,y = win_to_world(x,y)
        update_pos(int(4*x)/4,int(4*y)/4)


    pyg.app.run()

if __name__ == "__main__": test()
