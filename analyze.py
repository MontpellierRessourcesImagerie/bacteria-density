import tifffile as tiff
import numpy as np
from pprint import pprint
import os

from skimage.measure import label, regionprops
from skimage.restoration import rolling_ball
from scipy.ndimage import (gaussian_filter, affine_transform,
                           distance_transform_edt)
from skimage.filters import threshold_otsu
from skimage.morphology import (binary_closing, binary_opening, ball, 
                                skeletonize)
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import pandas as pd

# Where the croped input image is located.
image_path = "/home/clement/Downloads/2025-07-11-massilva/massilva-crop.tif"
# What channels are present, and which LUT to use for each one.
channels   = [('DAPI', 'cyan'), ('RFP', 'red')]
# Which channel should be used for the creation of the mask.
segment    = 'DAPI'
# On which channels should we measure.
measure    = ['RFP']
# Calibration
pxl_size   = (2.0, 0.325, 0.325)
# Size of bins for smoothing
bin_size   = 20 #um
# Downscaling factor for testing
scale      = (1.0, 1.0, 1.0)
# Folder in which the results will be written
output_dir = "/home/clement/Downloads/2025-07-11-massilva/output"

#########################################################

pxl_size = tuple([p * f for p, f in zip(pxl_size, scale)])

def get_image():
    """
    Helper function opening an image in a folder and splitting its channel.
    Each channel is then associated with a dye and a LUT.
    The global parameters 'image_path' and 'channels' are used.

    Returns:
        (dict) A dictionary like: {"dye-name": (channel_data, lut_name)}
    """
    raw_img = tiff.imread(image_path)
    data = {}
    scale_matrix = np.diag([1/f for f in scale])
    for idx, (ch_name, lut) in enumerate(channels):
        ch_data = raw_img[:,idx,:,:]
        output_shape = tuple(int(dim * f) for dim, f in zip(ch_data.shape, scale))
        rescaled = affine_transform(
            ch_data,
            scale_matrix,
            output_shape=output_shape,
            order=1 
        )
        print(f"Downsampling {ch_name}: {ch_data.shape} -> {rescaled.shape}")
        data[ch_name] = (rescaled, lut)
    return data

def keep_largest(mask):
    """
    Searches for the biggest connected component within a binary mask and isolates it.
    The input image is not modified.

    Args:
        - mask (np.ndarray): An image encoded on uint8 containing only 0 and 1.

    Returns:
        (np.ndarray)
    """
    labeled = label(mask)
    all_props = regionprops(labeled)
    biggest_size = -1
    biggest_lbl = 0
    for props in all_props:
        if int(props.label) == 0:
            continue
        if props.area > biggest_size:
            biggest_size = props.area
            biggest_lbl = props.label
    print(f"Found {len(all_props)} components, keeping ID-{biggest_lbl} ({biggest_size} voxels)")
    return labeled == biggest_lbl

def create_skeleton(nuclei_ch):
    print("Skeletonization...")
    nuclei_ch, lut = nuclei_ch
    print("  | [1/5] Gaussian smoothing")
    smoothed = gaussian_filter(nuclei_ch, 10.0)
    t = np.mean(smoothed)
    mask = (smoothed >= t).astype(np.uint8)
    k = ball(7)
    print("  | [2/5] Morphological closing")
    closed = binary_closing(mask, footprint=k)
    print("  | [3/5] Morphological opening")
    opened = binary_opening(closed, footprint=k)
    print("  | [4/5] Isolating the biggest chunk")
    largest = keep_largest(opened)
    print("  | [5/5] Creating the skeleton")
    skeleton = skeletonize(largest)
    print("  |    DONE.")
    return skeleton.astype(np.uint8), largest.astype(np.uint8)

def make_graph(skeleton):
    D, H, W = skeleton.shape
    shifts = [
        (-1, -1, -1),
        (-1, -1,  0),
        (-1, -1,  1),
        (-1,  0, -1),
        (-1,  0,  0),
        (-1,  0,  1),
        (-1,  1, -1),
        (-1,  1,  0),
        (-1,  1,  1),

        ( 0, -1, -1),
        ( 0, -1,  0),
        ( 0, -1,  1),
        ( 0,  0, -1),
        # ( 0,  0,  0),
        ( 0,  0,  1),
        ( 0,  1, -1),
        ( 0,  1,  0),
        ( 0,  1,  1),

        ( 1, -1, -1),
        ( 1, -1,  0),
        ( 1, -1,  1),
        ( 1,  0, -1),
        ( 1,  0,  0),
        ( 1,  0,  1),
        ( 1,  1, -1),
        ( 1,  1,  0),
        ( 1,  1,  1)
    ]
    graph = {}
    for pz, py, px in zip(*np.where(skeleton > 0)):
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
            if skeleton[z, y, x] > 0:
                nbrs.add( (int(z), int(y), int(x)) )
        graph[(int(pz), int(py), int(px))] = nbrs
    print("Graph built.")
    return graph

def get_roots(graph):
    return [k for k, v in graph.items() if len(v) == 1]

def longest_path_from(start, graph):
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
    return longest

def longest_path(roots, graph):
    print(f"Searching for longest path from {len(roots)} roots...")
    longest = []
    for r in roots:
        candidate = longest_path_from(r, graph)
        if len(candidate) > len(longest):
            longest = candidate
    print(f"  | Longest path found: {len(longest)} nodes")
    return longest

def process_normals(vertices, trail=8):
    npoints, dims = vertices.shape
    shp = (npoints-trail, dims)
    
    normals = np.zeros(shp, dtype=np.float32)
    basis = vertices[trail:].astype(np.float32)

    for i in range(0, trail):
        buffer = basis - vertices[i:npoints+i-trail].astype(np.float32)
        norm = np.linalg.norm(buffer, axis=1)
        buffer /= norm[:,np.newaxis]
        normals += buffer

    normals /= trail
    all_normals = np.zeros_like(vertices, dtype=np.float32)
    all_normals[trail:] = normals
    all_normals[:trail] = normals[0]
    print("Normals processed.")
    return all_normals

def make_vector(skeleton):
    graph = make_graph(skeleton)
    roots = sorted(get_roots(graph))
    vertices = np.array(longest_path(roots, graph))
    normals = process_normals(vertices)
    print("1D vector fully generated.")
    return vertices, normals

def as_rgb(normals):
    buffer = np.abs(normals)
    buffer *= 255
    return buffer[:, [2, 1, 0]].astype(np.uint8)

def make_control(skeleton, vertices, normals):
    D, H, W = skeleton.shape
    canvas = np.zeros((D, H, W, 3), dtype=np.uint8)
    rgb = as_rgb(normals)
    for i, (z, y, x) in enumerate(vertices):
        canvas[z, y, x] = rgb[i]
    return canvas

def process_width(vertices, mask):
    flat_z = np.max(mask, axis=0)
    dte = distance_transform_edt(flat_z) * pxl_size[2]
    vals = np.zeros((len(vertices),), dtype=np.float32)
    for i, (_, y, x) in enumerate(vertices):
        vals[i] = dte[y, x]
    print("Local width processed.")
    return vals

def process_intensity(vertices, mask, channel):
    intensities, lut = channel
    kd = KDTree(vertices)
    all_pts = np.where(mask > 0)
    all_pts = np.array([(z, y, x) for z, y, x in zip(*all_pts)])
    dist, indices = kd.query(all_pts)
    buffer = np.zeros((len(vertices), 2), dtype=np.float32)
    for i, (z, y, x) in zip(indices, all_pts):
        buffer[i] += (intensities[z, y, x], 1)
    print("Local density processed.")
    return buffer[:,0] / buffer[:,1]

def attribute_as_image(skeleton, vertices, values):
    img = np.zeros(skeleton.shape, dtype=np.float32)
    for i, (z, y, x) in enumerate(vertices):
        img[z, y, x] = values[i]
    return img

def cumulative_distance(vertices):
    diffs = np.diff(vertices, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    return cumulative

def bin_values(bin_length, distances, values):
    distances = np.asarray(distances)
    values = np.asarray(values)

    total_length = distances[-1]
    bins = np.arange(0, total_length + bin_length, bin_length)

    bin_indices = np.digitize(distances, bins) - 1

    binned_means = []
    binned_centers = []

    for i in range(len(bins) - 1):
        mask = bin_indices == i
        if np.any(mask):
            binned_means.append(values[mask].mean())
            binned_centers.append((bins[i] + bins[i+1]) / 2)

    return np.array(binned_centers), np.array(binned_means)

def save_as_csv(filename, distances, width, density):
    distances = np.asarray(distances)
    width = np.asarray(width)
    density = np.asarray(density)

    if not (len(distances) == len(width) == len(density)):
        raise ValueError("All input arrays must have the same length.")

    df = pd.DataFrame({
        "Distance (µm)": distances,
        "Width (µm)": width,
        "Density": density
    })

    df.to_csv(filename, index=False)

def save_plot(distances, values, binned_distances, binned_values, axis, title, output_path):
    plt.figure(figsize=(10, 6))

    plt.scatter(distances, values, alpha=0.05, s=10, label="Raw data")
    plt.plot(binned_distances, binned_values, color='orange', linewidth=2, label="Binned average")

    plt.title(title, fontsize=14)
    plt.xlabel(axis[0], fontsize=12)
    plt.ylabel(axis[1], fontsize=12)

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    data = get_image()
    skeleton, mask = create_skeleton(data['DAPI'])
    vertices, normals = make_vector(skeleton)
    ctrl = make_control(skeleton, vertices, normals)
    distances = cumulative_distance(vertices)

    attributes = {}
    attributes['width'] = process_width(vertices, mask)
    attributes['rfp'] = process_intensity(vertices, mask, data['RFP'])

    binned_distances, binned_width = bin_values(bin_size, distances, attributes['width'])
    binned_distances, binned_rfp = bin_values(bin_size, distances, attributes['rfp'])

    ctrl_width = attribute_as_image(skeleton, vertices, attributes['width'])
    ctrl_rfp = attribute_as_image(skeleton, vertices, attributes['rfp'])

    #################################################

    save_as_csv(os.path.join(output_dir, "raw-attributes.csv"), distances, attributes['width'], attributes['rfp'])
    save_as_csv(os.path.join(output_dir, "binned-attributes.csv"), binned_distances, binned_width, binned_rfp)

    tiff.imwrite(os.path.join(output_dir, "dapi.tif"), data['DAPI'][0])
    tiff.imwrite(os.path.join(output_dir, "rfp.tif"), data['RFP'][0])

    tiff.imwrite(os.path.join(output_dir, "mask.tif"), mask)
    tiff.imwrite(os.path.join(output_dir, "skeleton.tif"), skeleton)

    tiff.imwrite(os.path.join(output_dir, "control-width.tif"), ctrl_width)
    tiff.imwrite(os.path.join(output_dir, "control-rfp-density.tif"), ctrl_rfp)

    np.save(os.path.join(output_dir, "distances.npy"), distances)
    np.save(os.path.join(output_dir, "vertices.npy"), vertices)
    np.save(os.path.join(output_dir, "normals.npy"), normals)
    np.save(os.path.join(output_dir, "width.npy"), attributes['width'])
    np.save(os.path.join(output_dir, "rfp-density.npy"), attributes['rfp'])
    np.save(os.path.join(output_dir, "binned-width.npy"), binned_width)
    np.save(os.path.join(output_dir, "binned-rfp-density.npy"), binned_rfp)

    save_plot(
        distances, 
        attributes['rfp'], 
        binned_distances, 
        binned_rfp, 
        ("Distance (µm)", "Density"), 
        "Distance vs. density", 
        os.path.join(output_dir, "density-plot.png")
    )

    save_plot(
        distances, 
        attributes['width'], 
        binned_distances, 
        binned_width, 
        ("Distance (µm)", "Width (µm)"), 
        "Distance vs. width", 
        os.path.join(output_dir, "width-plot.png")
    )

if __name__ == "__main__":
    main()