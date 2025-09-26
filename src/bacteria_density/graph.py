import numpy as np

"""
This array defines the 26 possible 3D shifts to access all neighbors of a voxel in a 3D grid.
The center voxel (0, 0, 0) is excluded from this list.
"""
shifts = np.array([
    [-1, -1, -1],
    [-1, -1,  0],
    [-1, -1,  1],
    [-1,  0, -1],
    [-1,  0,  0],
    [-1,  0,  1],
    [-1,  1, -1],
    [-1,  1,  0],
    [-1,  1,  1],

    [ 0, -1, -1],
    [ 0, -1,  0],
    [ 0, -1,  1],
    [ 0,  0, -1],
    # [ ( 0,  0,  0),
    [ 0,  0,  1],
    [ 0,  1, -1],
    [ 0,  1,  0],
    [ 0,  1,  1],

    [ 1, -1, -1],
    [ 1, -1,  0],
    [ 1, -1,  1],
    [ 1,  0, -1],
    [ 1,  0,  0],
    [ 1,  0,  1],
    [ 1,  1, -1],
    [ 1,  1,  0],
    [ 1,  1,  1]
])

def skeleton_to_undirected_graph(skel):
    """
    Converts a 3D skeleton into an undirected graph representation.
    Each voxel in the skeleton is a node, and edges connect neighboring voxels.

    Args:
        - skel (np.ndarray): A binary 3D numpy array representing the skeleton.
    
    Returns:
        (dict): A dictionary where keys are voxel coordinates (z, y, x) and
                values are sets of neighboring voxel coordinates.
    """
    D, H, W = skel.shape
    graph = {}
    for pz, py, px in zip(*np.where(skel > 0)):
        nbrs = set() 
        for sz, sy, sx in shifts:
            z = pz + sz
            y = py + sy
            x = px + sx
            if (z < 0) or (z >= D):
                continue
            if (y < 0) or (y >= H):
                continue
            if (x < 0) or (x >= W):
                continue
            if skel[z, y, x] > 0:
                nbrs.add( (int(z), int(y), int(x)) )
        graph[(int(pz), int(py), int(px))] = nbrs
    return graph

def find_root(leaves, hint):
    """
    Among all leaf nodes, finds the one closest to the provided hint point.
    It will define the root of the graph traversal to find the longest path.

    Args:
        - leaves (list): A list of leaf nodes, where each node is a tuple (z, y, x).
        - hint (np.ndarray): A 2D point (y, x) to guide the root selection.

    Returns:
        (tuple): The leaf node closest to the hint point.
    """
    best_leaf = None
    best_dist = float('inf')
    for leaf in leaves:
        dist = np.linalg.norm(np.array(leaf[-2:]) - hint)
        if dist < best_dist:
            best_dist = dist
            best_leaf = leaf
    return best_leaf

def longest_path_from(start, graph, bbox):
    """
    Finds the longest path in an undirected graph starting from a given node.
    The graph is represented as a dictionary where keys are nodes and values are sets of neighboring nodes
    The returned path is a list of nodes representing the longest path found.
    In this path, nodes are in order from the root node and the node i+1 is the unique child of node i.

    Args:
        - start (tuple): The starting node (z, y, x) for the path search.
        - graph (dict): The undirected graph represented as a dictionary.

    Returns:
        (np.ndarray): An array of nodes representing the longest path found.
    """
    start = (start[0], start[1] - bbox[0], start[2] - bbox[1])
    stack = [start]
    visited = set()
    longest = []
    path = []
    while len(stack) > 0:
        current = stack.pop()
        if current in visited:
            continue
        path.append(current)
        visited.add(current)
        for e in graph[current]:
            stack.append(e)
        if len(graph[current]) == len(graph[current].intersection(visited)):
            if len(path) > len(longest):
                longest = path.copy()
            path.pop()
    return np.array(longest)

def get_leaves(bbox, graph):
    """
    Identifies leaf nodes in the graph, which are nodes with only one connection.
    """
    return [(k[0], k[1]+bbox[0], k[2]+bbox[1]) for k, v in graph.items() if len(v) == 1]

def find_next_bbox(hint, leaves, used_bboxes):
    best = float('inf')
    best_bbox = None
    root = None
    for leaf, bbox in leaves.items():
        if bbox in used_bboxes:
            continue
        dist = np.linalg.norm(leaf - hint)
        if (dist < best):
            best = dist
            best_bbox = bbox
            root = leaf
    return best_bbox, root