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
        self.t_factor        = 0.9
        self.merge_csv       = True
        self.to_process      = {k: False for k in measures.get_functions().keys()}
        self.processed       = set([])

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
                seg_crop = utils.make_crop(self.segmentation_ch, poly, bbox)
                tiff.imwrite(os.path.join(region_folder, "segmentation_ch.tif"), seg_crop)
                self.info(f"Exported segmentation channel for region '{id_str}' bbox {bbox}")
                for ch_name, ch_data in self.measurement_ch.items():
                    meas_crop = utils.make_crop(ch_data, poly, bbox)
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
    
    def _make_mask_and_skeleton(self, image, sigma=15.0, scale=0.5):
        print("    Downscaling image...")
        original_shape = image.shape
        scale_matrix = np.array([
            [1, 0, 0],
            [0, 1/scale, 0],
            [0, 0, 1/scale]
        ])
        new_shape = (original_shape[0], int(original_shape[1] * scale), int(original_shape[2] * scale))
        downscaled_image = affine_transform(image, scale_matrix, output_shape=new_shape, order=1)

        print("    Gaussian smoothing...")
        smoothed = gaussian_filter(downscaled_image, sigma)

        print("    Thresholding...")
        t = threshold_otsu(smoothed) * self.t_factor
        mask = (smoothed >= t).astype(np.uint8)

        print("    Morphological closing...")
        k = ball(7)
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
        largest = affine_transform(largest.astype(np.uint8), inv_scale_matrix, output_shape=original_shape, order=0)

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
                seg_crop = tiff.imread(seg_path)
                skeleton, mask = self._make_mask_and_skeleton(seg_crop)
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
    
    nuclei_path = "/home/clement/Documents/projects/2119-bacteria-density/small-data/nuclei.tif"
    bactos_path = "/home/clement/Documents/projects/2119-bacteria-density/small-data/bactos.tif"

    nuclei = tiff.imread(nuclei_path)
    bactos = tiff.imread(bactos_path)

    worker.set_working_dir("/home/clement/Documents/projects/2119-bacteria-density/small-data/output")

    worker.set_segmentation_channel(nuclei)
    worker.add_measurement_channel("RFP", bactos)

    worker.set_calibration((1.52, 0.325, 0.325), "µm")

    worker.add_region("id_001", 
        np.array([
            [   8.      ,  984.3847  ,  -24.18498 ],
            [   8.      , 1475.1365  ,  -29.793571],
            [   8.      , 1727.5231  ,  320.74338 ],
            [   8.      , 2361.294   ,  317.9391  ],
            [   8.      , 2832.4155  ,  472.17535 ],
            [   8.      , 2905.3271  , 1080.7075  ],
            [   8.      , 2386.5325  , 1686.4353  ],
            [   8.      , 1346.1388  , 1753.7384  ],
            [   8.      ,  922.6902  , 1770.5642  ],
            [   8.      ,  474.0029  , 1686.4353  ],
            [   8.      ,  782.4754  , 1321.877   ],
            [   8.      , 1298.4658  , 1215.3137  ],
            [   8.      , 1850.9121  , 1091.9247  ],
            [   8.      , 2055.6257  ,  791.86505 ],
            [   8.      , 1340.5303  ,  783.45215 ],
            [   8.      , 1006.8191  ,  200.15868 ]
        ])
    )

    worker.add_region("id_002", 
        np.array([
            [   8.       ,  485.2201   , 1807.02     ],
            [   8.       ,  510.45874  , 1792.9985   ],
            [   8.       ,  538.5017   , 1787.39     ],
            [   8.       ,  574.9575   , 1784.5857   ],
            [   8.       ,  614.21765  , 1784.5857   ],
            [   8.       ,  656.2821   , 1781.7814   ],
            [   8.       ,  726.3895   , 1781.7814   ],
            [   8.       ,  785.2797   , 1787.39     ],
            [   8.       ,  818.9313   , 1787.39     ],
            [   8.       ,  852.5828   , 1790.1943   ],
            [   8.       ,  953.5375   , 1807.02     ],
            [   8.       , 1037.6663   , 1832.2587   ],
            [   8.       , 1110.578    , 1871.5189   ],
            [   8.       , 1147.0338   , 1899.5618   ],
            [   8.       , 1177.8811   , 1916.3876   ],
            [   8.       , 1208.7284   , 1938.822    ],
            [   8.       , 1245.1842   , 1961.2563   ],
            [   8.       , 1276.0315   , 1989.2993   ],
            [   8.       , 1301.2701   , 2014.538    ],
            [   8.       , 1320.9001   , 2039.7766   ],
            [   8.       , 1351.7474   , 2076.2324   ],
            [   8.       , 1376.9861   , 2121.101    ],
            [   8.       , 1399.4204   , 2177.187    ],
            [   8.       , 1407.8334   , 2205.23     ],
            [   8.       , 1413.4419   , 2236.0774   ],
            [   8.       , 1416.2462   , 2264.1204   ],
            [   8.       , 1416.2462   , 2510.8982   ],
            [   8.       , 1421.8549   , 2552.9626   ],
            [   8.       , 1430.2677   , 2595.027    ],
            [   8.       , 1441.4849   , 2637.0916   ],
            [   8.       , 1461.115    , 2690.3733   ],
            [   8.       , 1477.9407   , 2743.6548   ],
            [   8.       , 1508.788    , 2808.1536   ],
            [   8.       , 1542.4396   , 2875.4568   ],
            [   8.       , 1578.8954   , 2942.7598   ],
            [   8.       , 1609.7427   , 3015.6714   ],
            [   8.       , 1646.1985   , 3085.7788   ],
            [   8.       , 1682.6543   , 3144.6692   ],
            [   8.       , 1716.3059   , 3195.1465   ],
            [   8.       , 1780.8047   , 3284.8838   ],
            [   8.       , 1825.6733   , 3357.7957   ],
            [   8.       , 1850.9121   , 3402.6643   ],
            [   8.       , 1867.7378   , 3441.9243   ],
            [   8.       , 1887.3679   , 3481.1846   ],
            [   8.       , 1901.3894   , 3517.6404   ],
            [   8.       , 1926.628    , 3573.7263   ],
            [   8.       , 1943.4539   , 3618.595    ],
            [   8.       , 1960.2795   , 3669.0723   ],
            [   8.       , 1974.301    , 3722.354    ],
            [   8.       , 1985.5182   , 3792.4614   ],
            [   8.       , 1991.1268   , 3837.33     ],
            [   8.       , 1999.5397   , 3879.3945   ],
            [   8.       , 2005.1483   , 3927.0676   ],
            [   8.       , 2007.9526   , 3974.7405   ],
            [   8.       , 2007.9526   , 4095.3252   ],
            [   8.       , 1996.7355   , 4142.9985   ],
            [   8.       , 1977.1053   , 4210.3013   ],
            [   8.       , 1963.0839   , 4243.953    ],
            [   8.       , 1951.8667   , 4274.8003   ],
            [   8.       , 1932.2366   , 4314.0605   ],
            [   8.       , 1915.4109   , 4342.1035   ],
            [   8.       , 1901.3894   , 4367.342    ],
            [   8.       , 1878.955    , 4400.9937   ],
            [   8.       , 1848.1078   , 4443.058    ],
            [   8.       , 1825.6733   , 4468.2964   ],
            [   8.       , 1797.6305   , 4504.7524   ],
            [   8.       , 1766.7832   , 4538.404    ],
            [   8.       , 1688.263    , 4611.316    ],
            [   8.       , 1665.8285   , 4633.75     ],
            [   8.       , 1595.7212   , 4689.836    ],
            [   8.       , 1567.6782   , 4706.6616   ],
            [   8.       , 1545.2438   , 4726.292    ],
            [   8.       , 1497.5708   , 4757.139    ],
            [   8.       , 1455.5063   , 4779.573    ],
            [   8.       , 1368.5732   , 4832.855    ],
            [   8.       , 1323.7045   , 4855.2896   ],
            [   8.       , 1270.4229   , 4877.7236   ],
            [   8.       , 1233.967    , 4894.5493   ],
            [   8.       , 1169.4683   , 4919.788    ],
            [   8.       , 1121.7952   , 4936.614    ],
            [   8.       , 1085.3394   , 4947.831    ],
            [   8.       , 1040.4706   , 4959.0483   ],
            [   8.       , 1001.21045  , 4967.4614   ],
            [   8.       ,  959.14606  , 4978.678    ],
            [   8.       ,  922.6902   , 4989.8955   ],
            [   8.       ,  880.6258   , 4998.3086   ],
            [   8.       ,  816.12695  , 5017.9385   ],
            [   8.       ,  737.6067   , 5043.1772   ],
            [   8.       ,  628.23914  , 5079.6333   ],
            [   8.       ,  549.7189   , 5102.0674   ],
            [   8.       ,  474.0029   , 5127.306    ],
            [   8.       ,  375.85254  , 5158.1533   ],
            [   8.       ,  345.0053   , 5169.3706   ],
            [   8.       ,  314.15805  , 5177.783    ],
            [   8.       ,  252.46355  , 5189.0005   ],
            [   8.       ,  193.57333  , 5194.6094   ],
            [   8.       ,   44.945667 , 5194.6094   ],
            [   8.       ,   16.90271  , 5189.0005   ],
            [   8.       ,  -19.553133 , 5169.3706   ],
            [   8.       ,  -50.400383 , 5132.9146   ],
            [   8.       ,  -67.22616  , 5102.0674   ],
            [   8.       , -103.682    , 5015.1343   ],
            [   8.       , -114.899185 , 4967.4614   ],
            [   8.       , -120.507774 , 4911.3755   ],
            [   8.       , -120.507774 , 4832.855    ],
            [   8.       , -123.31207  , 4793.5947   ],
            [   8.       , -123.31207  , 4748.726    ],
            [   8.       , -117.70348  , 4712.2705   ],
            [   8.       , -100.87771  , 4636.554    ],
            [   8.       ,  -95.26911  , 4600.0986   ],
            [   8.       ,  -75.639046 , 4513.1655   ],
            [   8.       ,  -64.42186  , 4471.101    ],
            [   8.       ,  -50.400383 , 4431.841    ],
            [   8.       ,  -27.966019 , 4364.5376   ],
            [   8.       ,   -5.5316544, 4305.6475   ],
            [   8.       ,   11.294119 , 4243.953    ],
            [   8.       ,   36.53278  , 4168.237    ],
            [   8.       ,   53.358555 , 4078.4995   ],
            [   8.       ,   75.792915 , 3971.9363   ],
            [   8.       ,  101.03158  , 3809.287    ],
            [   8.       ,  109.444466 , 3663.4639   ],
            [   8.       ,  120.66165  , 3526.0532   ],
            [   8.       ,  140.29172  , 3405.4685   ],
            [   8.       ,  168.33467  , 3169.9077   ],
            [   8.       ,  185.16045  , 2990.4329   ],
            [   8.       ,  201.98622  , 2869.8481   ],
            [   8.       ,  241.24637  , 2687.5688   ],
            [   8.       ,  288.91937  , 2496.8767   ],
            [   8.       ,  305.74515  , 2446.3994   ],
            [   8.       ,  316.96234  , 2415.5522   ],
            [   8.       ,  328.17953  , 2370.6836   ],
            [   8.       ,  336.5924   , 2331.4233   ],
            [   8.       ,  339.3967   , 2303.3804   ],
            [   8.       ,  345.0053   , 2264.1204   ],
            [   8.       ,  378.65686  , 2196.8171   ]
        ])
    )

    worker.add_region("id_003",
        np.array([
            [   8.      , -109.290596, -296.20166 ],
            [   8.      , -109.290596, 1187.27    ],
            [   8.      ,  689.93353 , 1187.27    ],
            [   8.      ,  689.93353 , -296.20166 ]
        ])
    )

    worker.set_starting_hint("id_001", [8.0, 1220.98960248, -6.11470609])
    worker.set_starting_hint("id_002", [8.0, 748.9657104  , 1817.38811913])
    worker.set_starting_hint("id_003", [8.0, 426.00199476 , -60.77010412])

    worker.use_metric("Density", True)
    worker.use_metric("Integrated intensity", True)
    worker.use_metric("Local width", True)
    worker.use_metric("Integrated volume", True)
    
    worker.export_regions()
    worker.make_mask_and_skeleton()
    worker.make_medial_path()
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
    # launch_analysis_workflow()
    # print("\n ============================ \n")
    launch_recover_workflow()