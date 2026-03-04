import numpy as np
from collections import deque

"""
This array defines the 26 possible 3D shifts to access all neighbors of a voxel in a 3D grid.
The center voxel (0, 0, 0) is excluded from this list.
"""
shifts = np.array([
    # direct (6-face) neighbors first
    [-1,  0,  0],
    [ 1,  0,  0],
    [ 0, -1,  0],
    [ 0,  1,  0],
    [ 0,  0, -1],
    [ 0,  0,  1],

    # edge (2-axis) diagonals
    [-1, -1,  0],
    [-1,  1,  0],
    [ 1, -1,  0],
    [ 1,  1,  0],

    [-1,  0, -1],
    [-1,  0,  1],
    [ 1,  0, -1],
    [ 1,  0,  1],

    [ 0, -1, -1],
    [ 0, -1,  1],
    [ 0,  1, -1],
    [ 0,  1,  1],

    # corner (3-axis) diagonals
    [-1, -1, -1],
    [-1, -1,  1],
    [-1,  1, -1],
    [-1,  1,  1],
    [ 1, -1, -1],
    [ 1, -1,  1],
    [ 1,  1, -1],
    [ 1,  1,  1],
], dtype=int)

def skeleton_to_undirected_graph(skel):
    """
    Converts a 3D skeleton into an undirected graph representation.
    First pass: create edges for direct (6-face) neighbors only.
    Second pass: add diagonal edges (edge and corner) only when the two
    voxels are not already connected via a path of direct neighbors.

    Args:
        - skel (np.ndarray): A binary 3D numpy array representing the skeleton.
    
    Returns:
        (dict): A dictionary where keys are voxel coordinates (z, y, x) and
                values are sets of neighboring voxel coordinates.
    """

    D, H, W = skel.shape
    # collect all skeleton nodes
    coords = [tuple(map(int, c)) for c in np.argwhere(skel > 0)]
    graph = {c: set() for c in coords}

    # direct (6-face) shifts are the first 6 entries of the global `shifts`
    direct_shifts = shifts[:6]
    diag_shifts = shifts[6:]

    # First pass: add direct neighbors
    for pz, py, px in coords:
        p = (pz, py, px)
        nbrs = set()
        for sz, sy, sx in direct_shifts:
            q = (pz + int(sz), py + int(sy), px + int(sx))
            if (0 <= q[0] < D) and (0 <= q[1] < H) and (0 <= q[2] < W):
                if skel[q] > 0:
                    nbrs.add(q)
        graph[p] = nbrs

    # Ensure symmetry for direct edges
    for p, nbrs in list(graph.items()):
        for q in nbrs:
            graph[q].add(p)

    # Use the direct-only graph for reachability tests
    direct_graph = {k: set(v) for k, v in graph.items()}

    def reachable_by_direct(start, target):
        """BFS on direct_graph to check if target is reachable from start."""
        if start == target:
            return True
        dq = deque([start])
        seen = {start}
        while dq:
            cur = dq.popleft()
            for nb in direct_graph.get(cur, ()):
                if nb == target:
                    return True
                if nb not in seen:
                    seen.add(nb)
                    dq.append(nb)
        return False

    # Second pass: consider diagonal neighbors but only add them if not reachable by direct path
    edges_to_add = []
    for p in coords:
        pz, py, px = p
        for sz, sy, sx in diag_shifts:
            q = (pz + int(sz), py + int(sy), px + int(sx))
            if not ((0 <= q[0] < D) and (0 <= q[1] < H) and (0 <= q[2] < W)):
                continue
            if skel[q] == 0:
                continue
            # if already a neighbor (shouldn't be for diag), skip
            if q in graph[p]:
                continue
            # if q cannot be reached from p using only direct neighbors, add diagonal edge
            if not reachable_by_direct(p, q):
                edges_to_add.append((p, q))

    # Apply diagonal edges (make undirected)
    for u, v in edges_to_add:
        graph[u].add(v)
        graph[v].add(u)

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

def longest_path_from(root, graph, bbox):
    neighbors = lambda v: graph.get(v, ())
    minx, miny, _, _ = bbox
    root = (root[0], root[1]-miny, root[2]-minx)

    path = [root]
    in_path = {root} 
    stack = [(root, iter(neighbors(root)))]  # (node, iterator over its neighbors)

    # Best-so-far
    best_path = list(path)

    while stack:
        node, it = stack[-1]
        try:
            nxt = next(it)
            if nxt in in_path:
                # Would create a cycle in the simple path; skip
                continue

            # Descend
            path.append(nxt)
            in_path.add(nxt)
            stack.append((nxt, iter(neighbors(nxt))))

            # Update best
            if len(path) > len(best_path):
                best_path = list(path)

        except StopIteration:
            # Backtrack
            stack.pop()
            # Pop the node from the current path only if this frame corresponds to it
            # (root is handled when its frame is popped)
            if path:
                last = path.pop()
                in_path.remove(last)

    best_path = np.array([(z, y, x) for (z, y, x) in best_path])
    return best_path


def get_leaves(bbox, graph):
    """
    Identifies leaf nodes in the graph, which are nodes with only one connection.
    """
    xmin, ymin, _, _ = bbox
    return [(k[0], k[1]+ymin, k[2]+xmin) for k, v in graph.items() if len(v) == 1]

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

def path_as_img(path):
    img = np.zeros((path[:, 1].max() + 1, path[:, 2].max() + 1), dtype=np.uint8)
    for z, y, x in path:
        img[y, x] = 255
    return img


def run1():
    import tifffile
    from pathlib import Path

    folder = Path("/home/clement/Documents/projects/2119-bacteria-density/09-02-2026/00-255-00/BB-832-16186-2105-17723")
    skel_path = folder / "skeleton.tif"
    skel = tifffile.imread(skel_path)
    skel = (skel > 0).astype(np.uint8)
    
    graph = skeleton_to_undirected_graph(skel)
    leaves = get_leaves((0, 0, 0, 0), graph)
    root = find_root(leaves, [(50, 900)])
    print("Root:", root)

    path1 = longest_path_from(root, graph, (0, 0, 0, 0))
    path2 = np.load(folder / "medial_path.npy")

    ctrl1 = path_as_img(path1)
    ctrl2 = path_as_img(path2)

    tifffile.imwrite(folder / "med-path-ctrl-1.tif", ctrl1)
    tifffile.imwrite(folder / "med-path-ctrl-2.tif", ctrl2)

if __name__ == "__main__":
    run1()