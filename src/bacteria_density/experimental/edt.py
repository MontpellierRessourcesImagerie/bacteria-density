import numpy as np
from scipy.ndimage import distance_transform_edt
import tifffile as tiff
from skimage.graph import route_through_array
from scipy.ndimage import gaussian_filter

def brightest_path(dt, start, end, fully_connected=True):
    """
    mask : 3D boolean ndarray (True = foreground/object)
    start, end : tuples (z,y,x) indices inside the mask
    sampling : voxel spacing (z,y,x) passed to distance_transform_edt
    smoothing_sigma : optional gaussian smoothing applied to the distance map
    fully_connected : if True, use 26-neighbour connectivity in 3D

    returns: dt (float ndarray), path (list of (z,y,x)), canvas (uint8 ndarray same shape as mask)
    """
    # 3) cost: prefer larger dt (brightest = cheapest)
    eps = 1e-9
    cost = dt.max() - dt + eps

    # 4) route through the 3D cost array
    path, path_cost = route_through_array(cost, start, end, fully_connected=fully_connected)

    # 5) draw path into canvas (same shape as mask). Use uint8 for easy saving/viewing.
    canvas = np.zeros_like(dt, dtype=np.uint8)
    for (z, y, x) in path:
        canvas[z, y, x] = 1

    return path, canvas


import numpy as np
from scipy.ndimage import distance_transform_edt

def dt_ignore_image_border_by_padding(mask,
                                      sampling=(1.0, 1.0, 1.0),
                                      pad_width=((0,0),(0,0),(0,0)),
                                      constant_values=((1,1),(1,1),(1,1))):
    """
    Robust padding + distance transform.

    mask : 3D boolean ndarray (True = foreground/object)
    sampling : tuple of floats (z,y,x) voxel spacing for distance_transform_edt
    pad_width : tuple of 3 pairs: ((z_before,z_after),(y_before,y_after),(x_before,x_after))
                or a scalar int will be converted to symmetric pad on all axes if you prefer.
    constant_values : tuple of 3 pairs: values to pad each axis with (use 0/1 or False/True)

    returns: dt (float ndarray) cropped to original mask.shape
    """
    mask = np.asarray(mask)
    if mask.ndim != 3:
        raise ValueError("mask must be 3D")

    # allow scalar pad_width for convenience
    if isinstance(pad_width, int):
        pad_width = ((pad_width, pad_width), (pad_width, pad_width), (pad_width, pad_width))

    # likewise allow terse constant_values like 0 or 1
    def normalize_const(cv):
        # if int or bool provided, broadcast to all axes
        if isinstance(cv, (int, bool)):
            return tuple(((int(cv), int(cv))) for _ in range(mask.ndim))
        # if single pair provided, broadcast to all axes
        if isinstance(cv, tuple) and len(cv) in (1,2) and all(isinstance(x, (int,bool)) for x in cv):
            return tuple(((int(cv[0]), int(cv[1])) if len(cv)==2 else (int(cv[0]), int(cv[0]))) for _ in range(mask.ndim))
        # if already per-axis pairs assume correct
        return cv

    constant_values = normalize_const(constant_values)

    # Pad using integer constants (0/1) then convert to boolean (safer dtype handling)
    padded_int = np.pad(mask.astype(np.uint8),
                        pad_width=pad_width,
                        mode='constant',
                        constant_values=constant_values)
    padded = padded_int.astype(bool)

    # compute distance transform (float)
    dt_padded = distance_transform_edt(padded, sampling=sampling)

    # crop back to original shape
    slices = tuple(slice(pad_width[ax][0], dt_padded.shape[ax] - pad_width[ax][1]) for ax in range(mask.ndim))
    dt = dt_padded[slices]
    return dt


in_path = "/tmp/exp-a-star/mask.tif"
out = "/tmp/exp-a-star/dt.tif"
mask3d = (tiff.imread(in_path) > 0).astype(np.uint8)
dt = dt_ignore_image_border_by_padding(mask3d)
tiff.imwrite(out, dt.astype(np.float32))

path, canvas = brightest_path(dt,
                                    (8, 2240, 0), 
                                    (8, 326, 938),
                                    fully_connected=True)
tiff.imwrite("/tmp/exp-a-star/path.tif", np.array(canvas, dtype=np.uint8)*255)