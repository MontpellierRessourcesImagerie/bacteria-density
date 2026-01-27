import termcolor
import tifffile as tiff
import numpy as np
import os
import json

import pandas as pd
from shapely.geometry import Point
from skimage.filters import threshold_otsu
from scipy.ndimage import (
    gaussian_filter,
    binary_fill_holes,
    affine_transform
)
from skimage.morphology import (
    binary_closing, 
    ball,
    binary_opening, 
    skeletonize
)
import bacteria_density.utils as utils
import bacteria_density.process as process
import bacteria_density.graph as graph
import bacteria_density.measure as measures
import bacteria_density.plots as plots

class BacteriaDensityWorker(object):
    
    def __init__(self):
        self.segmentation_ch = None
        self.segmentation_n  = "Nuclei"
        self.measurement_ch  = {}
        self.working_dir     = None
        self.chunks          = {}
        self.calibration     = (1.0, 1.0, 1.0)
        self.unit            = "pixel"
        self.binning_dist    = 10.0
        self.t_factor        = 0.7
        self.log_factor      = 150.0
        self.kernel_size     = 6
        self.merge_csv       = True
        self.to_process      = {k: False for k in measures.get_functions().keys()}
        self.processed       = set([])
        self.nuclei_range    = None

    def state_as_json(self):
        state = {}

    def _print(self, what, color, msg):
        print(termcolor.colored(f"[BacteriaDensity] ({what})", color, attrs=['bold']), msg)

    def info(self, msg):
        self._print("INFO", "blue", msg)

    def warning(self, msg):
        self._print("WARNING", "yellow", msg)

    def error(self, msg):
        self._print("ERROR", "red", msg)

    def set_calibration(self, calibration, unit="µm"):
        if not self.working_dir:
            self.error("Set working directory before setting calibration.")
            raise ValueError("Set working directory before setting calibration.")
        if len(calibration) != 3:
            raise ValueError("Calibration must be a tuple of three floats (z, y, x)")
        if any(c <= 0 for c in calibration):
            raise ValueError("Calibration values must be positive")
        self.calibration = calibration
        self.unit = unit
        calib_path = os.path.join(self.working_dir, "calibration.txt")
        with open(calib_path, "w") as f:
            f.write(f"{calibration[0]};{calibration[1]};{calibration[2]}\n")
            f.write(unit)
        self.info(f"Calibration set to {calibration} ({unit})")

    def recover_calibration(self):
        if self.working_dir is None:
            self.error("Set working directory before recovering calibration.")
            raise ValueError("Set working directory before recovering calibration.")
        calib_path = os.path.join(self.working_dir, "calibration.txt")
        if not os.path.isfile(calib_path):
            return False
        with open(calib_path, "r") as f:
            line1 = f.readline().strip()
            line2 = f.readline().strip()
        parts = line1.split(";")
        if len(parts) != 3:
            return False
        try:
            calibration = (float(parts[0]), float(parts[1]), float(parts[2]))
        except ValueError:
            return False
        if any(c <= 0 for c in calibration):
            return False
        unit = line2 if line2 else "pixel"
        self.set_calibration(calibration, unit)
        return True

    def save_metrics(self):
        if self.working_dir is None:
            self.error("Set working directory before saving metrics.")
            raise ValueError("Set working directory before saving metrics.")
        metrics_path = os.path.join(self.working_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(self.to_process, f, indent=2)
    
    def recover_metrics(self):
        if self.working_dir is None:
            self.error("Set working directory before recovering metrics.")
            raise ValueError("Set working directory before recovering metrics.")
        metrics_path = os.path.join(self.working_dir, "metrics.json")
        if not os.path.isfile(metrics_path):
            return False
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        for name, to_use in metrics.items():
            if name in self.to_process:
                self.use_metric(name, to_use)
        return True

    def use_metric(self, name, to_use):
        if name not in self.to_process:
            raise ValueError(f"Metric {name} not recognized")
        self.to_process[name] = to_use
        self.save_metrics()
        self.info(f"Metric '{name}' set to {to_use}")

    def set_binning_distance(self, dist):
        if dist <= 0:
            raise ValueError("Binning distance must be positive")
        self.binning_dist = dist
        self.info(f"Binning distance set to {dist} {self.unit}")

    def set_threshold_factor(self, factor):
        if factor <= 0:
            raise ValueError("Threshold factor must be positive")
        self.t_factor = factor
        self.info(f"Threshold factor set to {factor}")

    def set_log_factor(self, factor):
        if factor <= 0:
            raise ValueError("Log factor must be positive")
        self.log_factor = factor
        self.info(f"Log factor set to {factor}")

    def set_kernel_size(self, size):
        if size <= 0 or not isinstance(size, int):
            raise ValueError("Kernel size must be a positive integer")
        self.kernel_size = size
        self.info(f"Kernel size set to {size}")

    def save_original(self, name, image):
        if self.working_dir is None:
            self.error("Set working directory before saving original images.")
            raise ValueError("Set working directory before saving original images.")
        original_folder = os.path.join(self.working_dir, "originals")
        os.makedirs(original_folder, exist_ok=True)
        path = os.path.join(original_folder, f"{name}.tif")
        tiff.imwrite(path, image)

    def save_segmentation_ch(self):
        if self.working_dir is None:
            self.error("Set working directory before saving segmentation channel.")
            raise ValueError("Set working directory before saving segmentation channel.")
        seg_ch_path = os.path.join(self.working_dir, f"segmentation_ch.txt")
        with open(seg_ch_path, "w") as f:
            f.write(self.segmentation_n)

    def set_segmentation_channel(self, ch, name="Nuclei"):
        if self.working_dir is None:
            self.error("Set working directory before setting segmentation channel.")
            raise ValueError("Set working directory before setting segmentation channel.")
        d, h, w = ch.shape
        if d < 2 or h < 2 or w < 2:
            self.error("Segmentation channel must be a 3D image.")
            raise ValueError("Segmentation channel must be a 3D image.")
        self.segmentation_ch = ch
        self.segmentation_n = name
        self.save_original(self.segmentation_n, self.segmentation_ch)
        self.save_segmentation_ch()
        self.info(f"Segmentation channel set with shape {ch.shape}")
    
    def set_merge_measures(self, merge):
        self.merge_csv = merge
        self.info(f"Merge measures set to {merge}")

    def save_measurement_channels(self):
        if self.working_dir is None:
            self.error("Set working directory before saving segmentation channel.")
            raise ValueError("Set working directory before saving segmentation channel.")
        measures_ch_path = os.path.join(self.working_dir, f"measures.txt")
        with open(measures_ch_path, "w") as f:
            f.write(";".join(self.measurement_ch.keys()))

    def add_measurement_channel(self, name, ch):
        if self.working_dir is None:
            self.error("Set working directory before adding measurement channels.")
            raise ValueError("Set working directory before adding measurement channels.")
        if self.segmentation_ch is None:
            self.error("Set segmentation channel before adding measurement channels.")
            raise ValueError("Set segmentation channel before adding measurement channels.")
        if ch.shape != self.segmentation_ch.shape:
            self.error("Measurement channel shape must match segmentation channel shape.")
            raise ValueError("Measurement channel shape must match segmentation channel shape.")
        self.measurement_ch[name] = ch
        self.save_original(name, ch)
        self.save_measurement_channels()
        self.info(f"Measurement channel '{name}' added.")

    def set_working_dir(self, working_dir):
        if not os.path.isdir(working_dir):
            os.makedirs(working_dir, exist_ok=True)
        self.working_dir = working_dir
        self.info(f"Working directory set to {working_dir}")

    def chunks_as_json(self):
        as_json = {}
        for id_str, data in self.chunks.items():
            as_json[id_str] = {
                'start' : data['start'].tolist() if data['start'] is not None else None,
                'polygons': [[(0, y, x) for x, y in poly.exterior.coords] for poly in data['polygons']],
                'ranks' : {",".join(map(str, bbox)): rank for bbox, rank in data['ranks'].items()},
                # 'bboxes' : [list(bbox) for bbox in data['bboxes']]
            }
        return as_json

    def chunks_from_json(self, as_json):
        chunks = {}
        for id_str, data in as_json.items():
            chunks[id_str] = {
                'start' : np.array(data['start']) if data['start'] is not None else None,
                'polygons': [np.array(coords) for coords in data['polygons']],
                'ranks' : {tuple(map(int, bbox.split(","))): rank for bbox, rank in data['ranks'].items()},
                # 'bboxes' : [tuple(map(int, bbox.split(","))) for bbox in data['bboxes']]
            }
        return chunks

    def save_regions(self):
        if self.working_dir is None:
            raise ValueError("Working directory not set")
        regions_path = os.path.join(self.working_dir, "regions.json")
        with open(regions_path, "w") as f:
            as_json = json.dumps(self.chunks_as_json(), indent=2)
            f.write(as_json)
    
    def recover_regions(self):
        if self.working_dir is None:
            raise ValueError("Working directory not set")
        regions_path = os.path.join(self.working_dir, "regions.json")
        if not os.path.isfile(regions_path):
            return False
        chunks = {}
        with open(regions_path, "r") as f:
            chunks = json.load(f)
            chunks = self.chunks_from_json(chunks)
        for id_str, data in chunks.items(): # recovering the polygons
            for poly in data['polygons']:
                poly = np.array(poly)
                self.add_region(id_str, poly)
        for id_str, data in chunks.items(): # recovering starting points
            start = data['start']
            if start is not None:
                self.set_starting_hint(id_str, start)
        for id_str, data in chunks.items(): # recovering the ranks
            for bbox, rank in data['ranks'].items():
                bbox = tuple(bbox)
                if id_str in self.chunks and bbox in self.chunks[id_str]['ranks']:
                    self.chunks[id_str]['ranks'][bbox] = rank
        return True

    def add_region(self, id_str, coordinates):
        if self.segmentation_ch is None:
            raise ValueError("Segmentation channel not set")
        _, h, w = self.segmentation_ch.shape
        poly = utils.as_polygon(coordinates)
        (xmin, ymin, xmax, ymax) = poly.bounds
        ymin = int(round(ymin))
        xmin = int(round(xmin))
        ymax = int(round(ymax))
        xmax = int(round(xmax))
        ymin, xmin = max(0, ymin), max(0, xmin)
        ymax, xmax = min(h, ymax), min(w, xmax)
        if ymin >= ymax or xmin >= xmax:
            raise ValueError("Invalid bounding box coordinates")
        region_data = self.chunks.get(id_str, {
            'start'   : None,
            'polygons': [],
            'bboxes'  : [],
            'ranks'   : {}
        })
        bbox = (xmin, ymin, xmax, ymax)
        region_data['polygons'].append(poly)
        region_data['bboxes'].append(bbox)
        region_data['ranks'][bbox] = -1
        self.chunks[id_str] = region_data
        self.save_regions()
        self.info(f"Region '{id_str}' added with bbox {bbox}")

    def get_rank(self, id_str, bbox):
        if self.segmentation_ch is None:
            return -1
        _, h, w = self.segmentation_ch.shape
        (xmin, ymin, xmax, ymax) = bbox
        ymin = int(round(ymin))
        xmin = int(round(xmin))
        ymax = int(round(ymax))
        xmax = int(round(xmax))
        ymin, xmin = max(0, ymin), max(0, xmin)
        ymax, xmax = min(h, ymax), min(w, xmax)
        bbox = (xmin, ymin, xmax, ymax)
        if id_str not in self.chunks:
            raise ValueError(f"Region ID {id_str} not found.")
        if bbox not in self.chunks[id_str]['ranks']:
            raise ValueError(f"Bounding box {bbox} not found in region '{id_str}'.")
        return self.chunks[id_str]['ranks'].get(bbox, -1)

    def set_starting_hint(self, id_str, start):
        if id_str not in self.chunks:
            raise ValueError(f"Region ID {id_str} not found. Add region before setting starting hint.")
        polygons = self.chunks[id_str]['polygons']
        px = start[-1]
        py = start[-2]
        in_poly = [r for r, poly in enumerate(polygons) if poly.contains(Point(px, py))]
        if len(in_poly) != 1:
            raise ValueError("Starting hint must be within one of the region's polygons.")
        self.chunks[id_str]['start'] = np.array(start)
        self.save_regions()
        self.info(f"Starting hint for region '{id_str}' set to {self.chunks[id_str]['start']}.")

    def export_regions(self):
        if self.working_dir is None:
            raise ValueError("Working directory not set")
        if self.segmentation_ch is None:
            raise ValueError("Segmentation channel not set")
        for id_str, data in self.chunks.items():
            main_folder = os.path.join(self.working_dir, id_str)
            utils.reset_folder(main_folder)
            for poly, bbox in zip(data['polygons'], data['bboxes']):
                (xmin, ymin, xmax, ymax) = bbox
                region_folder = os.path.join(main_folder, utils.bbox_to_str((xmin, ymin, xmax, ymax)))
                utils.reset_folder(region_folder)
                self.nuclei_range = (np.min(self.segmentation_ch), np.max(self.segmentation_ch))
                seg_crop, poly_mask = utils.make_crop(self.segmentation_ch, poly, bbox)
                tiff.imwrite(os.path.join(region_folder, "segmentation_ch.tif"), seg_crop)
                tiff.imwrite(os.path.join(region_folder, "poly_mask.tif"), poly_mask)
                self.info(f"Exported segmentation channel for region '{id_str}' bbox {bbox}")
                for ch_name, ch_data in self.measurement_ch.items():
                    meas_crop, _ = utils.make_crop(ch_data, poly, bbox)
                    tiff.imwrite(os.path.join(region_folder, f"{ch_name}.tif"), meas_crop)
                    self.info(f"Exported measurement channel '{ch_name}' for region '{id_str}' bbox {bbox}")
    
    def are_regions_exported(self):
        if self.working_dir is None:
            raise ValueError("Working directory not set")
        if self.segmentation_ch is None:
            raise ValueError("Segmentation channel not set")
        for id_str, data in self.chunks.items():
            main_folder = os.path.join(self.working_dir, id_str)
            if not os.path.isdir(main_folder):
                return False
            for bbox in data['bboxes']:
                region_folder = os.path.join(main_folder, utils.bbox_to_str(bbox))
                seg_path = os.path.join(region_folder, "segmentation_ch.tif")
                if not os.path.isfile(seg_path):
                    return False
                for ch_name in self.measurement_ch.keys():
                    meas_path = os.path.join(region_folder, f"{ch_name}.tif")
                    if not os.path.isfile(meas_path):
                        return False
        return True
    
    def _make_mask_and_skeleton(self, image, poly_mask, sigma=8.0, scale=0.5):
        print("    Downscaling image...")
        original_shape = image.shape
        scale_matrix = np.array([
            [1,       0,       0],
            [0, 1/scale,       0],
            [0,       0, 1/scale]
        ])
        new_shape = (
            original_shape[0], 
            int(original_shape[1] * scale), 
            int(original_shape[2] * scale)
        )
        downscaled_image = affine_transform(
            image, 
            scale_matrix, 
            output_shape=new_shape, 
            order=1
        )

        print("    Gaussian smoothing...")
        smoothed = gaussian_filter(
            downscaled_image.astype(np.float32), 
            (sigma/2, sigma, sigma)
        )

        print("    Normalizing intensities...")
        if self.nuclei_range is None:
            raise ValueError("Nuclei intensity range not set")
        smoothed -= self.nuclei_range[0] * 1.5
        smoothed  = np.maximum(smoothed, 0.0)
        smoothed /= (self.nuclei_range[1] - self.nuclei_range[0])
        smoothed *= self.log_factor
        smoothed += 1.0
        smoothed = np.log(smoothed)

        print("    Thresholding...")
        t = threshold_otsu(smoothed) * self.t_factor
        # t = np.mean(smoothed) * self.t_factor
        mask = (smoothed >= t).astype(np.uint8)

        print("    Morphological closing...")
        k = ball(self.kernel_size)
        closed = binary_closing(mask, footprint=k)

        print("    Morphological opening...")
        opened = binary_opening(closed, footprint=k)

        print("    Filling holes...")
        for i in range(opened.shape[0]):
            opened[i] = binary_fill_holes(opened[i])
        
        print("    Keeping largest component...")
        largest = process.keep_largest(opened)

        print("    Upscaling mask back to original size...")
        inv_scale_matrix = np.array([
            [1, 0, 0],
            [0, scale, 0],
            [0, 0, scale]
        ])
        largest = affine_transform(
            largest.astype(np.uint8), 
            inv_scale_matrix, 
            output_shape=original_shape, 
            order=0
        )

        print("    Multiply each slice by the mask polygon...")
        poly_mask = (poly_mask > 0).astype(np.uint8)
        for i in range(largest.shape[0]):
            largest[i] *= poly_mask
        
        print("    Skeletonization...")
        skeleton = skeletonize(largest)

        print("    Mask and skeleton generated.")
        return skeleton.astype(np.uint8), largest.astype(np.uint8)

    def make_mask_and_skeleton(self):
        if self.working_dir is None:
            raise ValueError("Working directory not set")
        for id_str, data in self.chunks.items():
            main_folder = os.path.join(self.working_dir, id_str)
            for bbox in data['bboxes']:
                region_folder = os.path.join(main_folder, utils.bbox_to_str(bbox))
                seg_path = os.path.join(region_folder, "segmentation_ch.tif")
                if not os.path.isfile(seg_path):
                    self.error(f"Segmentation channel file not found for region '{id_str}' bbox {bbox}")
                    continue
                mask_path = os.path.join(region_folder, "poly_mask.tif")
                if not os.path.isfile(mask_path):
                    self.error(f"Poly mask file not found for region '{id_str}' bbox {bbox}")
                    continue
                seg_crop = tiff.imread(seg_path)
                poly_mask = tiff.imread(mask_path)
                skeleton, mask = self._make_mask_and_skeleton(seg_crop, poly_mask)
                tiff.imwrite(os.path.join(region_folder, "mask.tif"), mask)
                tiff.imwrite(os.path.join(region_folder, "skeleton.tif"), skeleton)
                self.info(f"Generated mask and skeleton for region '{id_str}' bbox {bbox}")
    
    def are_skeletons_generated(self):
        if self.working_dir is None:
            raise ValueError("Working directory not set")
        for id_str, data in self.chunks.items():
            main_folder = os.path.join(self.working_dir, id_str)
            for bbox in data['bboxes']:
                region_folder = os.path.join(main_folder, utils.bbox_to_str(bbox))
                skel_path = os.path.join(region_folder, "skeleton.tif")
                mask_path = os.path.join(region_folder, "mask.tif")
                if not os.path.isfile(skel_path) or not os.path.isfile(mask_path):
                    return False
        return True
    
    def _make_graphs(self, main_folder, bboxes, u_graphs, leaves):
        for bbox in bboxes:
            region_folder = os.path.join(main_folder, utils.bbox_to_str(bbox))
            skel_path = os.path.join(region_folder, "skeleton.tif")
            if not os.path.isfile(skel_path):
                self.error(f"Skeleton file not found for {bbox}. Generate skeleton before medial path.")
                continue
            skeleton = tiff.imread(skel_path)
            if np.sum(skeleton) == 0:
                self.error(f"Skeleton is empty for {bbox}. Medial path not generated.")
                continue
            u_graphs[bbox] = graph.skeleton_to_undirected_graph(skeleton)
            leaves.update({l: bbox for l in graph.get_leaves(bbox, u_graphs[bbox])})
    
    def are_medial_paths_generated(self):
        if self.working_dir is None:
            raise ValueError("Working directory not set")
        for id_str, data in self.chunks.items():
            main_folder = os.path.join(self.working_dir, id_str)
            for bbox in data['bboxes']:
                region_folder = os.path.join(main_folder, utils.bbox_to_str(bbox))
                med_path_file = os.path.join(region_folder, "medial_path.npy")
                if not os.path.isfile(med_path_file):
                    return False
        return True

    def make_medial_path(self):
        if self.working_dir is None:
            raise ValueError("Working directory not set")
        for id_str, data in self.chunks.items():
            main_folder = os.path.join(self.working_dir, id_str)
            u_graphs = {} # bbox: graph, in local coordinates
            leaves   = {} # leaf: bbox, in global coordinates
            self._make_graphs(main_folder, data['bboxes'], u_graphs, leaves)
            hint = data['start'] # in global coordinates
            used_bboxes = set([])
            for rank in range(len(data['bboxes'])):
                bbox, root = graph.find_next_bbox(hint, leaves, used_bboxes)
                if (bbox is None) or (root is None):
                    self.error(f"No more unprocessed bounding box for region '{id_str}'. Medial path not generated.")
                    break
                used_bboxes.add(bbox)
                self.chunks[id_str]['ranks'][bbox] = rank
                med_path = graph.longest_path_from(root, u_graphs[bbox], bbox)
                if med_path is None or len(med_path) == 0:
                    self.error(f"Could not find a path in bbox {bbox} for region '{id_str}'.")
                    continue
                hint = np.array([med_path[-1][0], med_path[-1][1] + bbox[1], med_path[-1][2] + bbox[0]]) # in global coordinates
                region_folder = os.path.join(main_folder, utils.bbox_to_str(bbox))
                np.save(os.path.join(region_folder, "medial_path.npy"), med_path)
                self.save_regions()
                self.info(f"Generated medial path for region '{id_str}' bbox {bbox}")
    
    def are_measures_generated(self):
        if self.working_dir is None:
            raise ValueError("Working directory not set")
        metrics_path = os.path.join(self.working_dir, "processed_metrics.json")
        if not os.path.isfile(metrics_path):
            return False
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        self.processed = set(metrics)
        binning_path = os.path.join(self.working_dir, "binning_distance.txt")
        if not os.path.isfile(binning_path):
            return False
        with open(binning_path, "r") as f:
            line = f.readline().strip()
        try:
            binning_dist = float(line)
        except ValueError:
            return False
        self.set_binning_distance(binning_dist)
        for id_str, data in self.chunks.items():
            main_folder = os.path.join(self.working_dir, id_str)
            for bbox in data['bboxes']:
                region_folder = os.path.join(main_folder, utils.bbox_to_str(bbox))
                results_path = os.path.join(region_folder, "results.csv")
                if not os.path.isfile(results_path):
                    return False
                ctrl_met_folder = os.path.join(region_folder, "ctrl-metrics")
                if not os.path.isdir(ctrl_met_folder):
                    return False
        return True

    def _make_measures(self, mask, med_path, affectations, region_folder):
        results_table = {}
        cumulative_dists = measures.cumulative_distance(med_path, self.calibration)
        binned_dists, bins_indices = utils.get_binned_distances(self.binning_dist, cumulative_dists)
        processed = set([])
        
        for when in measures.get_execution_order():
            measure_fx = measures.get_metrics_by_order(when, self.to_process)
            for ch_name in self.measurement_ch.keys():
                ch_path = os.path.join(region_folder, f"{ch_name}.tif")
                ch_data = tiff.imread(ch_path)
                for metric, func in measure_fx.items():
                    if when == 'U': # Unique for the patch
                        if metric in results_table:
                            continue
                        else:
                            result = func(None, mask, self.calibration, len(med_path), affectations, med_path, bins_indices, self.binning_dist)
                            results_table[metric] = result
                            processed.add(metric)
                            process.make_metric_control(result, bins_indices, med_path, mask.shape, metric, region_folder)
                            self.info(f"Computed metric [U]: '{metric}'.")
                    elif when == 'C': # To do for each channel
                        result = func(ch_data, mask, self.calibration, len(med_path), affectations, med_path, bins_indices, self.binning_dist)
                        results_table[f"{ch_name} - {metric}"] = result
                        processed.add(f"{ch_name} - {metric}")
                        process.make_metric_control(result, bins_indices, med_path, mask.shape, f"{ch_name} - {metric}", region_folder)
                        self.info(f"Computed metric [C]: '{metric}' for channel '{ch_name}'.")
                    elif when == 'P':
                        result = func(ch_name, results_table)
                        process.make_metric_control(result, bins_indices, med_path, mask.shape, f"{ch_name} - {metric}", region_folder)
                        results_table[f"{ch_name} - {metric}"] = result
                        processed.add(f"{ch_name} - {metric}")
                        self.info(f"Computed metric [P]: '{metric}' for channel '{ch_name}'.")
        
        results_table["Cumulative distance"] = binned_dists[:-1]
        processed.add("Cumulative distance")
        process.make_metric_control(binned_dists[:-1], bins_indices, med_path, mask.shape, "Cumulative distance", region_folder)
        self.processed = processed
        return results_table
    
    def get_processed_metrics(self):
        return list(self.processed)
    
    def merge_measures(self):
        if self.working_dir is None:
            raise ValueError("Working directory not set")
        if len(self.chunks) == 0:
            raise ValueError("No regions to merge")
        for id_str, data in self.chunks.items():
            all_dfs = []
            main_folder = os.path.join(self.working_dir, id_str)
            # Sort bounding boxes by their rank before merging
            sorted_bboxes = sorted(data['bboxes'], key=lambda bbox: data['ranks'].get(bbox, -1))
            for bbox in sorted_bboxes:
                region_folder = os.path.join(main_folder, utils.bbox_to_str(bbox))
                results_path = os.path.join(region_folder, "results.csv")
                if not os.path.isfile(results_path):
                    self.error(f"Results file not found for region '{id_str}' bbox {bbox}. Cannot merge.")
                    continue
                df = pd.read_csv(results_path)
                # Add a row of NaN to mark the split between regions
                df = pd.concat([df, pd.DataFrame([np.nan] * len(df.columns)).T.set_axis(df.columns, axis=1)], ignore_index=True)
                all_dfs.append(df)
            if len(all_dfs) == 0:
                self.error(f"No results files found to merge for region '{id_str}'.")
                continue
            merged_df = pd.concat(all_dfs, ignore_index=True)
            merged_path = os.path.join(main_folder, f"{id_str}_merged_results.csv")
            merged_df.to_csv(merged_path, index=False)
            self.info(f"Merged results for region '{id_str}' saved to {merged_path}")
    
    def make_plots(self):
        if self.working_dir is None:
            raise ValueError("Working directory not set")
        if len(self.chunks) == 0:
            raise ValueError("No regions to plot")
        for id_str, _ in self.chunks.items():
            main_folder = os.path.join(self.working_dir, id_str)
            merged_path = os.path.join(main_folder, f"{id_str}_merged_results.csv")
            if not os.path.isfile(merged_path):
                self.error(f"Merged results file not found for region '{id_str}'. Cannot create plots.")
                continue
            plots.main(merged_path, out_dir=os.path.join(main_folder, "plots"), x_column="Cumulative distance")
            self.info(f"Plots generated for region '{id_str}' in {os.path.join(main_folder, 'plots')}")

    def export_processed_set(self):
        if self.working_dir is None:
            raise ValueError("Working directory not set")
        processed_path = os.path.join(self.working_dir, "processed_metrics.json")
        with open(processed_path, "w") as f:
            json.dump(list(self.processed), f, indent=2)
        binning_path = os.path.join(self.working_dir, "binning_distance.txt")
        with open(binning_path, "w") as f:
            f.write(f"{self.binning_dist}\n")

    def get_segmentation_channel(self):
        return self.segmentation_n, self.segmentation_ch
    
    def get_measurement_channels(self):
        return self.measurement_ch

    def make_measures(self):
        if self.working_dir is None:
            raise ValueError("Working directory not set")
        for id_str, data in self.chunks.items():
            self.info(f"Starting measures for region: {id_str}")
            main_folder = os.path.join(self.working_dir, id_str)
            for bbox in data['bboxes']:
                region_folder = os.path.join(main_folder, utils.bbox_to_str(bbox))
                mask_path = os.path.join(region_folder, "mask.tif")
                if not os.path.isfile(mask_path):
                    self.error(f"Mask file not found for region '{id_str}' bbox {bbox}. Generate mask before measures.")
                    continue
                mask = tiff.imread(mask_path)

                med_path_file = os.path.join(region_folder, "medial_path.npy")
                if not os.path.isfile(med_path_file):
                    self.error(f"Medial path file not found for region '{id_str}' bbox {bbox}. Generate medial path before measures.")
                    continue
                med_path = np.load(med_path_file)

                affectations = process.make_affectations(mask, med_path)
                affectations_ctrl_path = os.path.join(region_folder, "affectations_control.tif")
                control = process.control_affectations(mask, affectations)
                tiff.imwrite(affectations_ctrl_path, control)

                results_table = self._make_measures(mask, med_path, affectations, region_folder)
                results_path = os.path.join(region_folder, "results.csv")
                df = pd.DataFrame(results_table)
                df.to_csv(results_path, index=False)
                self.info(f"Performed measurements for region '{id_str}' bbox {bbox}.")
        self.export_processed_set()
        if self.merge_csv:
            self.merge_measures()
    
    def recover_images(self):
        if self.working_dir is None:
            raise ValueError("Working directory not set")
        original_folder = os.path.join(self.working_dir, "originals")
        if not os.path.isdir(original_folder):
            return False
        seg_ch_path = os.path.join(self.working_dir, f"segmentation_ch.txt")
        if not os.path.isfile(seg_ch_path):
            return False
        with open(seg_ch_path, "r") as f:
            seg_name = f.read().strip()
        seg_path = os.path.join(original_folder, f"{seg_name}.tif")
        if not os.path.isfile(seg_path):
            return False
        self.set_segmentation_channel(tiff.imread(seg_path), seg_name)
        measures_ch_path = os.path.join(self.working_dir, f"measures.txt")
        if not os.path.isfile(measures_ch_path):
            return False
        with open(measures_ch_path, "r") as f:
            measure_names = f.read().strip().split(";")
        for name in measure_names:
            ch_path = os.path.join(original_folder, f"{name}.tif")
            if not os.path.isfile(ch_path):
                return False
            self.add_measurement_channel(name, tiff.imread(ch_path))
        return True

    def recover(self):
        if self.working_dir is None:
            raise ValueError("Working directory not set")
        if not self.recover_images():
            return False
        if not self.recover_calibration():
            return False
        if not self.recover_regions():
            return False
        if not self.recover_metrics():
            return False
        if not self.are_regions_exported():
            return False
        print("Exported regions detected.")
        if not self.are_skeletons_generated():
            return False
        print("Masks & skeletons detected.")
        if not self.are_medial_paths_generated():
            return False
        print("Medial paths detected.")
        if not self.are_measures_generated():
            return False
        print("Measurements detected.")
        return True


def launch_analysis_workflow():
    worker = BacteriaDensityWorker()
    
    nuclei_path = "/home/clement/Downloads/bacteria-testing-crops/Nuclei.tif"
    bactos_path = "/home/clement/Downloads/bacteria-testing-crops/Rick.tif"

    nuclei = tiff.imread(nuclei_path)
    bactos = tiff.imread(bactos_path)

    worker.set_working_dir("/home/clement/Downloads/bacteria-testing-crops/wd")

    worker.set_segmentation_channel(nuclei)
    worker.add_measurement_channel("RICK", bactos)
    
    worker.add_region("id_001", 
        np.array([
            [   6.     ,  845.9978 , 1561.8286 ],
            [   6.     , 1462.5442 , 1432.6123 ],
            [   6.     , 2603.3396 , 1129.877  ],
            [   6.     , 3371.2537 ,  742.228  ],
            [   6.     , 4142.8594 ,  546.5577 ],
            [   6.     , 4441.903  ,  734.8443 ],
            [   6.     , 4312.6865 ,  889.9038 ],
            [   6.     , 4113.324  ,  827.1416 ],
            [   6.     , 3928.7297 ,  996.96875],
            [   6.     , 3278.9563 , 1355.0825 ],
            [   6.     , 2481.507  , 2093.4614 ],
            [   6.     , 1311.1765 , 2252.213  ],
            [   6.     ,  683.55444, 1912.5586 ]
        ])
    )

    worker.add_region("id_002", 
        np.array([
            [6.0000000e+00, 5.3314751e+03, 1.3887511e+03],
            [6.0000000e+00, 5.5446821e+03, 1.5234081e+03],
            [6.0000000e+00, 5.8532710e+03, 1.5290188e+03],
            [6.0000000e+00, 6.9866333e+03, 1.2540941e+03],
            [6.0000000e+00, 8.4229746e+03, 9.0061969e+02],
            [6.0000000e+00, 9.6629404e+03, 7.5474133e+02],
            [6.0000000e+00, 1.1458366e+04, 6.0886298e+02],
            [6.0000000e+00, 1.2400965e+04, 6.1447369e+02],
            [6.0000000e+00, 1.2462683e+04, 7.8840558e+02],
            [6.0000000e+00, 1.2451461e+04, 1.0352766e+03],
            [6.0000000e+00, 1.1654741e+04, 1.4111940e+03],
            [6.0000000e+00, 9.9490859e+03, 1.5514615e+03],
            [6.0000000e+00, 8.6137383e+03, 1.9217682e+03],
            [6.0000000e+00, 6.9080835e+03, 2.1630286e+03],
            [6.0000000e+00, 5.5222393e+03, 1.9329896e+03],
            [6.0000000e+00, 5.1912075e+03, 1.5682937e+03]
        ])
    )

    worker.add_region("id_002", 
        np.array([
            [6.00000000e+00, 1.26246621e+04, 6.08350708e+02],
            [6.00000000e+00, 1.24363750e+04, 8.07713013e+02],
            [6.00000000e+00, 1.24289912e+04, 9.40621216e+02],
            [6.00000000e+00, 1.28314082e+04, 1.17321057e+03],
            [6.00000000e+00, 1.42527871e+04, 1.07722131e+03],
            [6.00000000e+00, 1.54120420e+04, 1.71961096e+03],
            [6.00000000e+00, 1.57184697e+04, 1.74545422e+03],
            [6.00000000e+00, 1.57332373e+04, 1.64946497e+03],
            [6.00000000e+00, 1.56593994e+04, 1.59777844e+03],
            [6.00000000e+00, 1.53972744e+04, 1.21751331e+03],
            [6.00000000e+00, 1.49874746e+04, 8.85242798e+02],
            [6.00000000e+00, 1.44299980e+04, 4.68058716e+02],
            [6.00000000e+00, 1.35107168e+04, 4.45907349e+02],
            [6.00000000e+00, 1.29089375e+04, 4.12680298e+02]
        ])
    )

    worker.add_region("id_002", 
        np.array([
            [6.0000000e+00, 1.5684020e+04, 1.6843350e+03],
            [6.0000000e+00, 1.5691800e+04, 1.8269684e+03],
            [6.0000000e+00, 1.5927793e+04, 2.1031587e+03],
            [6.0000000e+00, 1.6270113e+04, 2.2756155e+03],
            [6.0000000e+00, 1.6782297e+04, 2.4260288e+03],
            [6.0000000e+00, 1.7107762e+04, 2.7320425e+03],
            [6.0000000e+00, 1.7230943e+04, 2.1368721e+03],
            [6.0000000e+00, 1.7212791e+04, 1.9410752e+03],
            [6.0000000e+00, 1.7059785e+04, 1.9657119e+03],
            [6.0000000e+00, 1.6954754e+04, 2.0214686e+03],
            [6.0000000e+00, 1.6879547e+04, 2.0499954e+03],
            [6.0000000e+00, 1.6787484e+04, 2.0655552e+03],
            [6.0000000e+00, 1.6729135e+04, 2.0940820e+03],
            [6.0000000e+00, 1.6642258e+04, 2.0901919e+03],
            [6.0000000e+00, 1.6535930e+04, 2.0344353e+03],
            [6.0000000e+00, 1.6472395e+04, 2.0033152e+03],
            [6.0000000e+00, 1.6382924e+04, 2.0097986e+03],
            [6.0000000e+00, 1.6280487e+04, 1.9994252e+03],
            [6.0000000e+00, 1.6270113e+04, 2.0409186e+03],
            [6.0000000e+00, 1.6307717e+04, 2.0564785e+03],
            [6.0000000e+00, 1.6384221e+04, 2.0616653e+03],
            [6.0000000e+00, 1.6455537e+04, 2.0811152e+03],
            [6.0000000e+00, 1.6541117e+04, 2.1083452e+03],
            [6.0000000e+00, 1.6612434e+04, 2.1303887e+03],
            [6.0000000e+00, 1.6681158e+04, 2.1485420e+03],
            [6.0000000e+00, 1.6753771e+04, 2.1433552e+03],
            [6.0000000e+00, 1.6810824e+04, 2.1264985e+03],
            [6.0000000e+00, 1.6884734e+04, 2.1329819e+03],
            [6.0000000e+00, 1.6949568e+04, 2.1342786e+03],
            [6.0000000e+00, 1.7011807e+04, 2.0850054e+03],
            [6.0000000e+00, 1.7071455e+04, 2.0668521e+03],
            [6.0000000e+00, 1.7014400e+04, 2.1316853e+03],
            [6.0000000e+00, 1.7002730e+04, 2.1978154e+03],
            [6.0000000e+00, 1.7024773e+04, 2.2639453e+03],
            [6.0000000e+00, 1.7031258e+04, 2.3534155e+03],
            [6.0000000e+00, 1.7029961e+04, 2.4428855e+03],
            [6.0000000e+00, 1.7015697e+04, 2.3884255e+03],
            [6.0000000e+00, 1.7000137e+04, 2.2937688e+03],
            [6.0000000e+00, 1.6963830e+04, 2.1939253e+03],
            [6.0000000e+00, 1.6893811e+04, 2.1511353e+03],
            [6.0000000e+00, 1.6800451e+04, 2.1381687e+03],
            [6.0000000e+00, 1.6720057e+04, 2.1511353e+03],
            [6.0000000e+00, 1.6672080e+04, 2.1563220e+03],
            [6.0000000e+00, 1.6568348e+04, 2.1277954e+03],
            [6.0000000e+00, 1.6487953e+04, 2.1031587e+03],
            [6.0000000e+00, 1.6402373e+04, 2.0811152e+03],
            [6.0000000e+00, 1.6323277e+04, 2.0707419e+03],
            [6.0000000e+00, 1.6271410e+04, 2.0551819e+03],
            [6.0000000e+00, 1.6227323e+04, 2.0072052e+03],
            [6.0000000e+00, 1.6106733e+04, 1.9345918e+03],
            [6.0000000e+00, 1.5990033e+04, 1.8710552e+03],
            [6.0000000e+00, 1.5905750e+04, 1.7802883e+03],
            [6.0000000e+00, 1.5838323e+04, 1.7050817e+03],
            [6.0000000e+00, 1.5763116e+04, 1.6674783e+03]
        ])
    )

    worker.add_region("id_002", 
        np.array([
            [6.00000000e+00, 1.72037148e+04, 1.87753845e+03],
            [6.00000000e+00, 1.71025742e+04, 1.93199854e+03],
            [6.00000000e+00, 1.70247734e+04, 1.96960181e+03],
            [6.00000000e+00, 1.69275234e+04, 2.02017188e+03],
            [6.00000000e+00, 1.68289766e+04, 2.04999536e+03],
            [6.00000000e+00, 1.67226504e+04, 2.08500537e+03],
            [6.00000000e+00, 1.66202129e+04, 2.07722534e+03],
            [6.00000000e+00, 1.65268535e+04, 2.03313855e+03],
            [6.00000000e+00, 1.64529434e+04, 2.00590857e+03],
            [6.00000000e+00, 1.63647705e+04, 2.00720520e+03],
            [6.00000000e+00, 1.62895635e+04, 1.93459180e+03],
            [6.00000000e+00, 1.62765967e+04, 1.86068176e+03],
            [6.00000000e+00, 1.62247305e+04, 1.83604504e+03],
            [6.00000000e+00, 1.62364004e+04, 1.73490503e+03],
            [6.00000000e+00, 1.63868145e+04, 1.74138831e+03],
            [6.00000000e+00, 1.66357734e+04, 1.84771509e+03],
            [6.00000000e+00, 1.70364434e+04, 1.72712500e+03],
            [6.00000000e+00, 1.71181348e+04, 1.34849805e+03],
            [6.00000000e+00, 1.71700000e+04, 1.06971448e+03],
            [6.00000000e+00, 1.73670938e+04, 1.09564783e+03],
            [6.00000000e+00, 1.73982148e+04, 1.19160132e+03],
            [6.00000000e+00, 1.74008086e+04, 1.04507776e+03],
            [6.00000000e+00, 1.73178203e+04, 9.54311096e+02],
            [6.00000000e+00, 1.75175078e+04, 8.59654358e+02],
            [6.00000000e+00, 1.76290215e+04, 9.38751038e+02],
            [6.00000000e+00, 1.76329121e+04, 1.12158118e+03],
            [6.00000000e+00, 1.75058379e+04, 1.18382129e+03],
            [6.00000000e+00, 1.76458789e+04, 1.27199463e+03],
            [6.00000000e+00, 1.76355039e+04, 1.48853821e+03],
            [6.00000000e+00, 1.73826543e+04, 1.51576819e+03],
            [6.00000000e+00, 1.73411602e+04, 1.38221143e+03],
            [6.00000000e+00, 1.73178203e+04, 1.51706482e+03],
            [6.00000000e+00, 1.73165234e+04, 1.70637830e+03],
            [6.00000000e+00, 1.72568770e+04, 1.83085840e+03]
        ])
    )

    worker.set_starting_hint("id_001", [   6.        , 4310.40854068,  800.9030318 ])
    worker.set_starting_hint("id_002", [   6.        , 5289.57477442, 1539.68783768])

    worker.use_metric("Density", True)
    worker.use_metric("Integrated intensity", True)
    worker.use_metric("Local width", True)
    worker.use_metric("Integrated volume", True)
    
    worker.export_regions()
    worker.make_mask_and_skeleton()
    worker.make_medial_path()
    return
    worker.make_measures()
    worker.make_plots()

def launch_synthetic_workflow():
    worker = BacteriaDensityWorker()
    
    nuclei_path = "/home/clement/Documents/projects/2119-bacteria-density/small-data/nuclei-1.tif"
    bactos_path = "/home/clement/Documents/projects/2119-bacteria-density/small-data/bactos-1.tif"

    nuclei = tiff.imread(nuclei_path)
    bactos = tiff.imread(bactos_path)

    worker.set_working_dir("/home/clement/Documents/projects/2119-bacteria-density/small-data/output-1")

    worker.set_segmentation_channel(nuclei)
    worker.add_measurement_channel("RFP", bactos)

    worker.add_region("id_001",
        np.array([
            [   2.      ,  272.2688  ,  502.82126 ],
            [   2.      ,  270.86417 ,  140.42494 ],
            [   2.      ,  352.3331  ,  -43.582493],
            [   2.      ,  539.14984 ,  -49.201042],
            [   2.      ,  567.24255 ,  735.99097 ],
            [   2.      ,  543.3637  , 1476.2346  ],
            [   2.      ,  393.06757 , 1613.889   ],
            [   2.      ,  241.36679 , 1508.5413  ],
            [   2.      ,  172.53958 ,  977.5885  ],
            [   2.      ,  178.15814 ,  669.973   ]
        ])
    )

    worker.set_starting_hint("id_001", [  2.        , 416.24408502,  15.41225465])

    worker.use_metric("Density", True)
    worker.use_metric("Integrated intensity", True)
    worker.use_metric("Local width", True)
    worker.use_metric("Integrated volume", True)
    
    worker.export_regions()
    worker.make_mask_and_skeleton()
    worker.make_medial_path()
    worker.make_measures()
    worker.make_plots()

def launch_recover_workflow():
    worker = BacteriaDensityWorker()
    worker.set_working_dir("/home/clement/Documents/projects/2119-bacteria-density/small-data/output")
    worker.recover()

if __name__ == "__main__":
    # launch_synthetic_workflow()
    launch_analysis_workflow()
    # print("\n ============================ \n")
    # launch_recover_workflow()