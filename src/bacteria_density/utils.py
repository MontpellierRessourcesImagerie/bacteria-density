import numpy as np
import random
import os
import shutil
from shapely.geometry import Polygon
from rasterio import features
import tifffile
from PIL import Image, ImageDraw

def as_polygon(coordinates):
    xy = coordinates[..., -2:][..., ::-1]
    return Polygon(xy)

def from_polygon(polygon):
    return np.array(polygon.exterior.coords)[..., ::-1]

def polygon_to_mask(poly, shape):
    height, width = shape
    if poly.is_empty:
        return np.zeros(shape, dtype=bool)

    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)

    def _draw_single_polygon(p):
        exterior = [(x, y) for x, y in p.exterior.coords]
        draw.polygon(exterior, outline=1, fill=1)
        for interior in p.interiors:
            hole = [(x, y) for x, y in interior.coords]
            draw.polygon(hole, outline=0, fill=0)

    if poly.geom_type == "Polygon":
        _draw_single_polygon(poly)
    elif poly.geom_type == "MultiPolygon":
        for p in poly.geoms:
            _draw_single_polygon(p)
    else:
        raise TypeError(f"Unsupported geometry type: {poly.geom_type}")

    mask = np.array(img, dtype=bool)
    return mask

def make_crop(image, shape, bbox):
    """
    Crop a (N,...,H,W) image stack to the polygon `shape`, safely clamped to the image
    bounds and optionally intersected with `bbox`.

    Args:
        image: array-like where last two dims are (H, W) (e.g. (Z, H, W) or (C, H, W))
        shape: a shapely Polygon in the same pixel coordinate space as the image
        bbox: optional tuple (ymin, xmin, ymax, xmax) to further intersect the crop

    Returns:
        The cropped stack as an array with the same leading dims as `image` and
        spatial dims reduced to the intersection (may be empty if no overlap).
    """
    H, W = image.shape[-2:]
    # rasterize polygon into image-sized mask
    mask2d = features.rasterize(
        [(shape, 1)],
        out_shape=(H, W),
        fill=0,
        dtype=image.dtype
    )

    # apply mask to image (broadcasts over leading dims)
    masked = mask2d[np.newaxis, ...] * image
    xmin, ymin, xmax, ymax = bbox
    return masked[:, ymin:ymax, xmin:xmax]

def get_binned_distances(bin_length, distances):
    total_length = distances[-1]
    binned_distances = np.arange(0, total_length + bin_length, bin_length)
    bin_indices = np.searchsorted(distances, binned_distances, side='left')
    bin_indices = np.clip(bin_indices, 0, len(distances) - 1)
    return binned_distances, bin_indices

def sum_in_bins(bin_indices, values):
    sums = []
    for i in range(len(bin_indices) - 1):
        start = bin_indices[i]
        end = bin_indices[i + 1]
        sums.append(np.sum(values[start:end]))
    return np.array(sums)

def avg_in_bins(bin_indices, values):
    avgs = []
    for i in range(len(bin_indices) - 1):
        start = bin_indices[i]
        end = bin_indices[i + 1]
        segment = values[start:end]
        if len(segment) == 0:
            avgs.append(0)
        else:
            avgs.append(np.mean(segment))
    return np.array(avgs)

def random_id():
    return hex(random.getrandbits(64))[2:].zfill(16)

def polygon_to_bbox(polygon):
    """
    Converts a Napari polygon to a 2D bounding box.
    """
    min_coords = np.min(polygon, axis=0)[-2:]
    max_coords = np.max(polygon, axis=0)[-2:]
    return (min_coords[0], min_coords[1], max_coords[0], max_coords[1])

def bbox_to_polygon(bbox):
    """
    Converts a 2D bounding box to a Napari polygon.
    """
    (ymin, xmin, ymax, xmax) = bbox
    return np.array([
        [ymin, xmin],
        [ymin, xmax],
        [ymax, xmax],
        [ymax, xmin]
    ])

def clr_to_str(clr):
    complement = [1 for _ in range(3 - len(clr))]
    color = list(clr) + complement
    color = color[:3]
    return "-".join([str(int(round(c, 1) * 255)).zfill(2) for c in color])

def str_to_clr(clr_str):
    parts = clr_str.split("-")
    if len(parts) not in [3, 4]:
        raise ValueError("Color string must have three or four components")
    return tuple(int(p) / 255.0 for p in parts)

def bbox_to_str(bbox):
    return "BB-" + "-".join([str(int(c)) for c in bbox])

def reset_folder(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
