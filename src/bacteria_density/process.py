import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import dilation
from scipy.spatial import KDTree
import tifffile as tiff
import os

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
    return np.asarray(labeled == biggest_lbl, dtype=np.uint8)

def make_affectations(mask, path):
    kd = KDTree(path)
    all_pts = np.where(mask > 0)
    all_pts = np.array(list(zip(*all_pts)))
    _, indices = kd.query(all_pts)
    affectations = {}
    for i, (z, y, x) in zip(indices, all_pts):
        affectations[(z, y, x)] = i
    return affectations

def control_affectations(mask, affectations):
    control = np.zeros_like(mask, dtype=np.uint32)
    for (z, y, x), i in affectations.items():
        control[z, y, x] = i + 1
    return control

def make_metric_control(values, bins_indices, med_path, shape, metric, region_folder):
    control = np.zeros(shape, dtype=np.float32)
    for i in range(len(bins_indices) - 1):
        start = bins_indices[i]
        end = bins_indices[i + 1]
        for j in range(start, end):
            z, y, x = med_path[j]
            control[z, y, x] = values[i]
    control = dilation(control, np.ones((3,7,7), dtype=np.uint8))
    output_folder = os.path.join(region_folder, "ctrl-metrics")
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"{metric.replace(' ', '_').lower()}.tif")
    tiff.imwrite(output_path, control)