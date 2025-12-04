import numpy as np
from scipy.ndimage import distance_transform_edt

import bacteria_density.utils as utils

"""
In this module, all functions must take the exact same input so they can be called blindly in a loop.

Args:
    - image_data (np.ndarray): A 3D numpy array containing the image.
    - calibration (tuple): A tuple of three floats representing the voxel size in each dimension (z, y, x).
    - n_bins (int): Number of bins to use for histogram-based measures.
    - affectations (np.ndarray): A dictionary in which keys are 3D coordinates and the values are the index in which the voxel belongs.

Returns:
    - (np.ndarray): A 1D numpy array of length 'n_bins' containing the computed measure for each region.
"""

def integrated_intensity(image_data, mask, calibration, n_bins, affectations, med_path, bins_indices, binning_dist):
    buffer = np.zeros(n_bins, dtype=np.uint32)
    for (z, y, x), i in affectations.items():
        buffer[i] += image_data[z, y, x]
    buffer = utils.sum_in_bins(bins_indices, buffer)
    return buffer

def integrated_volume(image_data, mask, calibration, n_bins, affectations, med_path, bins_indices, binning_dist):
    buffer = np.zeros(n_bins, dtype=np.float32)
    unit_volume = calibration[0] * calibration[1] * calibration[2]
    for _, i in affectations.items():
        buffer[i] += 1
    buffer *= unit_volume
    buffer = utils.sum_in_bins(bins_indices, buffer)
    return buffer

def local_width(image_data, mask, calibration, n_bins, affectations, med_path, bins_indices, binning_dist):
    buffer = np.zeros(n_bins, dtype=np.float32)
    r = distance_transform_edt(mask, sampling=calibration)
    dte = np.asarray(r)
    if dte is None:
        return
    for i, (z, y, x) in enumerate(med_path):
        buffer[i] = max(dte[z, y, x], buffer[i])
    buffer = utils.avg_in_bins(bins_indices, buffer)
    return buffer

def local_density(ch_name, results_table):
    integrated_volume = results_table.get("Integrated volume", None)
    if integrated_volume is None:
        return
    integrated_intensity = results_table.get(f"{ch_name} - Integrated intensity", None)
    if integrated_intensity is None:
        return
    density = integrated_intensity / (integrated_volume + 1)
    return density

#####################################################################################

def cumulative_distance(med_path, calibration):
    diffs = np.diff(med_path, axis=0)
    diffs_calibrated = diffs * np.array(calibration)
    segment_lengths = np.linalg.norm(diffs_calibrated, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    return cumulative

# U: To be done only once per image
# C: To be done for each channel
# P: To be processed from other measures

def get_execution_order():
    return ['U', 'C', 'P']

def get_functions():
    return {
        "Integrated intensity": (integrated_intensity, 'C'),
        "Integrated volume"   : (integrated_volume   , 'U'),
        "Local width"         : (local_width         , 'U'),
        "Density"             : (local_density       , 'P')
    }

def get_metrics_by_order(target_when, to_process=None):
    fx = get_functions()
    uniq_metrics = {}
    for metric, (f, when) in fx.items():
        if when != target_when:
            continue
        if (to_process is None) or (to_process.get(metric, False)):
            uniq_metrics[metric] = f
    return uniq_metrics
