import os
from qtpy.QtWidgets import (QWidget, QVBoxLayout, 
                            QGroupBox, QHBoxLayout, QLabel, 
                            QComboBox, QCheckBox, QLineEdit, 
                            QPushButton, QFileDialog, QDoubleSpinBox
)
from qtpy.QtCore import Qt, QThread

import napari
from napari.utils import progress

import tifffile
import numpy as np
from shapely.geometry import Point

from bacteria_density.utils import polygon_to_bbox, bbox_to_str, clr_to_str
import bacteria_density.measure as measures
import bacteria_density.utils as utils
from bacteria_density.bd_worker import BacteriaDensityWorker

from bacteria_density.qt_workers import (QtExportCrops, QtMakeMasks, 
                                         QtMedialAxis, QtMeasure, QtPlot)

NEUTRAL = "--------"
colors_rgba = [
    (1.0, 0.0, 0.0, 1.0), # red
    (0.0, 1.0, 0.0, 1.0), # green
    (0.0, 0.0, 1.0, 1.0), # blue
    (1.0, 1.0, 0.0, 1.0), # yellow
    (1.0, 0.0, 1.0, 1.0), # magenta
    (0.0, 1.0, 1.0, 1.0), # cyan
    (1.0, 0.5, 0.0, 1.0), # orange
    (0.5, 0.0, 1.0, 1.0), # purple
]

class BacteriaDensityWidget(QWidget):

    def __init__(self, viewer=None, parent=None):
        super().__init__(parent)
        viewer = viewer or napari.current_viewer()
        if viewer is None:
            raise ValueError("No Napari viewer instance found.")
        self.viewer = viewer

        # Channels on which intensity measures will be performed
        self.measure_rows    = [] # list of dicts: {'label': QLabel, 'combo': QComboBox, 'edit': QLineEdit, 'container': QWidget}
        self.selected_folder = None # Folder in which controls will be exported
        self.channel_pools   = set([]) # List of combo boxes containing layer names, to be refreshed when layers change
        self.image_areas     = {} # Each color is a track with a collection of bboxes and a hint for starter.
        self.model           = BacteriaDensityWorker()

        self._build_ui()
        self.refresh_layer_names()

        self.viewer.layers.events.inserted.connect(lambda e: self.refresh_layer_names())
        self.viewer.layers.events.removed.connect(lambda e: self.refresh_layer_names())
        self.viewer.layers.events.reordered.connect(lambda e: self.refresh_layer_names())

        self.qt_thread = None
        self.qt_worker = None
        self.pbr       = None

    # ---------------- UI BUILD ----------------

    def _build_ui(self):
        root = QVBoxLayout(self)

        # Segmentation
        self.gb_seg = QGroupBox("Segmentation")
        seg_l = QHBoxLayout()
        seg_l.addWidget(QLabel("Nuclei channel:"))
        self.cb_seg_channel = QComboBox()
        self.channel_pools.add(self.cb_seg_channel)
        seg_l.addWidget(self.cb_seg_channel, 1)
        self.gb_seg.setLayout(seg_l)
        root.addWidget(self.gb_seg)

        # Measures
        self.gb_meas = QGroupBox("Measures")
        self.meas_v = QVBoxLayout()
        self.gb_meas.setLayout(self.meas_v)
        root.addWidget(self.gb_meas)

        # start with a single empty row
        self._add_measure_row()

        # ROIs
        self.gb_rois = QGroupBox("ROIs")
        rois_v = QVBoxLayout()

        h_layout = QHBoxLayout()
        for i in range(1, 5):
            button = QPushButton(f"F{i}")
            button.clicked.connect(lambda _, c=i: self.set_shape_filament(c))
            h_layout.addWidget(button)
        rois_v.addLayout(h_layout)

        self.cb_roi_start = self._make_layer_row(rois_v, "Starting points:")
        self.cb_roi_poly  = self._make_layer_row(rois_v, "Bounding polygons:")

        self.gb_rois.setLayout(rois_v)
        root.addWidget(self.gb_rois)

        # Settings
        self.gb_settings = QGroupBox("Settings")
        set_v = QVBoxLayout()

        # folder button
        self.btn_pick_folder = QPushButton("Set output folder")
        self.btn_pick_folder.clicked.connect(self._pick_folder)
        set_v.addWidget(self.btn_pick_folder)

        # line: float input + "µm" label (label left-aligned)
        line = QHBoxLayout()
        bin_label = QLabel("Binning length:")
        line.addWidget(bin_label)
        self.sb_pixel = QDoubleSpinBox()
        self.sb_pixel.setDecimals(1)
        self.sb_pixel.setRange(1.0, 100.0)
        self.sb_pixel.setSingleStep(0.1)
        self.sb_pixel.setValue(10.0)
        line.addWidget(self.sb_pixel)
        um_label = QLabel("µm")
        line.addWidget(um_label)
        set_v.addLayout(line)

        # Checkboxes for each metric
        set_v.addSpacing(10)
        self._add_metrics_checkboxes(set_v)

        self.gb_settings.setLayout(set_v)
        root.addWidget(self.gb_settings)

        # Workflow
        self.gb_workflow = QGroupBox("Workflow")
        set_v = QVBoxLayout()

        self.btn_chunks = QPushButton("Chunk images")
        self.btn_chunks.clicked.connect(self.export_crops)
        set_v.addWidget(self.btn_chunks)

        mask_line = QHBoxLayout()
        self.sb_mask_factor = QDoubleSpinBox()
        self.sb_mask_factor.setPrefix("x")
        self.sb_mask_factor.setDecimals(2)
        self.sb_mask_factor.setRange(0.01, 10.0)
        self.sb_mask_factor.setSingleStep(0.01)
        self.sb_mask_factor.setValue(1.00)
        mask_line.addWidget(self.sb_mask_factor)
        self.btn_mask = QPushButton("Skeletonize")
        self.btn_mask.clicked.connect(self._make_masks)
        mask_line.addWidget(self.btn_mask)
        set_v.addLayout(mask_line)

        self.btn_skeletons = QPushButton("Make medial path")
        self.btn_skeletons.clicked.connect(self._make_medial_path)
        set_v.addWidget(self.btn_skeletons)

        h_layout = QHBoxLayout()

        # Add a checkbox to merge measures
        self.cb_merge_measures = QCheckBox("Merge measures")
        self.cb_merge_measures.setChecked(True)
        h_layout.addWidget(self.cb_merge_measures)

        self.btn_metrics = QPushButton("Measure")
        self.btn_metrics.clicked.connect(self._make_measures)
        h_layout.addWidget(self.btn_metrics)

        set_v.addLayout(h_layout)

        # add a combo box to visualize metrics
        self.cb_visualize = QComboBox()
        self.cb_visualize.addItems([NEUTRAL])
        self.cb_visualize.currentIndexChanged.connect(self._show_control)
        set_v.addWidget(self.cb_visualize)

        # add a button to create the plots
        self.btn_create_plots = QPushButton("Create Plots")
        self.btn_create_plots.clicked.connect(self._create_plots)
        set_v.addWidget(self.btn_create_plots)

        self.gb_workflow.setLayout(set_v)
        root.addWidget(self.gb_workflow)

        root.addStretch(1)
        self.setLayout(root)

    def set_shape_filament(self, filament_class):
        l = self.viewer.layers.selection.active
        if l is None:
            return
        if not hasattr(l, 'edge_color'):
            return
        if len(l.edge_color) == 0:
            return
        color = colors_rgba[(filament_class - 1) % len(colors_rgba)]
        all_edges = l.edge_color
        all_edges[-1] = color
        l.edge_width = 5
        l.edge_color = all_edges
        l.face_color = 'transparent'

    def set_activated_ui(self, active):
        self.gb_seg.setEnabled(active)
        self.gb_meas.setEnabled(active)
        self.gb_rois.setEnabled(active)
        self.gb_settings.setEnabled(active)
        self.gb_workflow.setEnabled(active)
        for row in self.measure_rows:
            row['combo'].setEnabled(active)
            row['edit'].setEnabled(active)
        for cb in self.metrics_checkboxes.values():
            cb.setEnabled(active)
        if not active:
            self.pbr = progress(total=0)
        else:
            if self.pbr is not None:
                self.pbr.close()
                self.pbr = None

    def _make_layer_row(self, parent_layout, label_text):
        row = QHBoxLayout()
        row.addWidget(QLabel(label_text))
        cb = QComboBox()
        self.channel_pools.add(cb)
        row.addWidget(cb, 1)
        parent_layout.addLayout(row)
        return cb

    def _add_metrics_checkboxes(self, parent):
        main_layout = QHBoxLayout()
        col1 = QVBoxLayout()
        col2 = QVBoxLayout()

        names = measures.get_functions().keys()

        self.metrics_checkboxes = {}
        for i, name in enumerate(names):
            cb = QCheckBox(name)
            cb.setCheckState(Qt.CheckState.Checked)
            self.metrics_checkboxes[name] = cb
            if i % 2 == 0:
                col1.addWidget(cb)
            else:
                col2.addWidget(cb)

        col1.addStretch()
        col2.addStretch()

        main_layout.addLayout(col1)
        main_layout.addLayout(col2)
        parent.addLayout(main_layout)

    # ---------------- MEASURES DYNAMIC ----------------

    def _make_masks(self):
        # Set the mask factor
        mask_factor = self.sb_mask_factor.value()
        self.model.set_threshold_factor(mask_factor)
        self.qt_worker = QtMakeMasks(self.model)
        self.qt_thread = None
        self.qt_thread = QThread()
        self.qt_worker.moveToThread(self.qt_thread)
        self.qt_thread.started.connect(self.qt_worker.run)
        self.qt_worker.finished.connect(self.qt_thread.quit)
        self.qt_worker.finished.connect(self.qt_worker.deleteLater)
        self.qt_thread.finished.connect(self._finished_make_masks)
        self.qt_thread.start()
        self.set_activated_ui(False)

    def _finished_make_masks(self):
        print("Finished making masks.")
        self.set_activated_ui(True)
        if self.model.working_dir is None:
            return
        if self.model.segmentation_ch is None:
            return
        for clr_key, collection in self.model.chunks.items():
            for bbox in collection['bboxes']:
                mask_path = os.path.join(self.model.working_dir, clr_key, bbox_to_str(bbox), "mask.tif")
                if not os.path.isfile(mask_path):
                    continue
                mask_name = f"mask-{bbox_to_str(bbox)}"
                layer = None
                data = tifffile.imread(mask_path)
                if mask_name in self.viewer.layers:
                    layer = self.viewer.layers[mask_name]
                    layer.data = data
                else:
                    layer = self.viewer.add_labels(data, name=mask_name)
                layer.scale = self.model.calibration
                xmin, ymin, _, _ = bbox
                layer.translate = np.array([0, ymin, xmin]) * self.model.calibration
                layer.contour = 6

    def _make_medial_path(self):
        self.qt_worker = QtMedialAxis(self.model)
        self.qt_thread = None
        self.qt_thread = QThread()
        self.qt_worker.moveToThread(self.qt_thread)
        self.qt_thread.started.connect(self.qt_worker.run)
        self.qt_worker.finished.connect(self.qt_thread.quit)
        self.qt_worker.finished.connect(self.qt_worker.deleteLater)
        self.qt_thread.finished.connect(self._finished_make_medial_path)
        self.qt_thread.start()
        self.set_activated_ui(False)
    
    def _finished_make_medial_path(self):
        self.set_activated_ui(True)
        print("Finished making medial paths.")
        if self.model.working_dir is None:
            return
        paths = []
        for clr_key, collection in self.model.chunks.items():
            for bbox in collection['bboxes']:
                path_path = os.path.join(self.model.working_dir, clr_key, bbox_to_str(bbox), "medial_path.npy")
                if not os.path.isfile(path_path):
                    continue
                path_name = f"medial-{bbox_to_str(bbox)}"
                data = np.load(path_path)
                paths.append(data)
                layer = None
                if path_name in self.viewer.layers:
                    layer = self.viewer.layers[path_name]
                    layer.data = data
                else:
                    layer = self.viewer.add_shapes(data, name=path_name, shape_type='path')
                layer.scale = self.model.calibration
                xmin, ymin, _, _ = bbox
                layer.translate = np.array([0, ymin, xmin]) * self.model.calibration
                layer.edge_color = np.array(utils.str_to_clr(clr_key))
                layer.edge_width = 5
        self._show_rank()

    def _make_measures(self):
        # Set the binning distance
        binning = self.sb_pixel.value()
        self.model.set_binning_distance(binning)
        # Set the metrics to compute
        for name, cb in self.metrics_checkboxes.items():
            self.model.use_metric(name, cb.isChecked())
        self.model.set_merge_measures(self.cb_merge_measures.isChecked())
        self.qt_worker = QtMeasure(self.model)
        self.qt_thread = None
        self.qt_thread = QThread()
        self.qt_worker.moveToThread(self.qt_thread)
        self.qt_thread.started.connect(self.qt_worker.run)
        self.qt_worker.finished.connect(self.qt_thread.quit)
        self.qt_worker.finished.connect(self.qt_worker.deleteLater)
        self.qt_thread.finished.connect(self._finished_make_measures)
        self.qt_thread.start()
        self.set_activated_ui(False)

    def _finished_make_measures(self):
        print("Finished making measures.")
        self.set_activated_ui(True)
        self.cb_visualize.clear()
        self.cb_visualize.addItems([NEUTRAL] + list(self.model.get_processed_metrics()))

    def _show_control(self):
        if self.model.working_dir is None:
            return
        if self.model.segmentation_ch is None:
            return
        sel = self.cb_visualize.currentText()
        if sel == NEUTRAL:
            return
        as_file_name = sel.replace(" ", "_").lower() + ".tif"
        layers = []
        low, up = float('inf'), float('-inf')
        for clr_key, collection in self.model.chunks.items():
            for bbox in collection['bboxes']:
                ctrl_path = os.path.join(self.model.working_dir, clr_key, bbox_to_str(bbox), "ctrl-metrics", as_file_name)
                if not os.path.isfile(ctrl_path):
                    continue
                ctrl_name = f"ctrl-{bbox_to_str(bbox)}"
                layer = None
                data = tifffile.imread(ctrl_path)
                low = min(low, np.nanmin(data))
                up  = max(up , np.nanmax(data))
                if ctrl_name in self.viewer.layers:
                    layer = self.viewer.layers[ctrl_name]
                    layer.data = data
                else:
                    layer = self.viewer.add_image(data, name=ctrl_name)
                layer.scale = self.model.calibration
                xmin, ymin, _, _ = bbox
                layer.translate = np.array([0, ymin, xmin]) * self.model.calibration
                layer.blending = "additive"
                layer.colormap = "magma"
                layers.append(layer)
        for l in layers:
            l.contrast_limits = (low, up)

    def export_crops(self):
        # Set the output folder
        if self.selected_folder is None:
            self.model.error("No output folder selected.")
            return
        if not os.path.isdir(self.selected_folder):
            self.model.error(f"Output folder '{self.selected_folder}' does not exist.")
            return
        self.model.set_working_dir(self.selected_folder)
        # set the segmentation channel
        seg_layer_name = self.cb_seg_channel.currentText()
        if seg_layer_name in self.viewer.layers:
            self.model.set_segmentation_channel(self.viewer.layers[seg_layer_name].data)
        else:
            self.model.error(f"Segmentation layer '{seg_layer_name}' not found.")
            return
        # Set the measurement channels
        measures_config = self.get_measures_config()
        for m in measures_config:
            layer_name = m['layer']
            if layer_name in self.viewer.layers:
                self.model.add_measurement_channel(m['alias'], self.viewer.layers[layer_name].data)
            else:
                self.model.error(f"Measure layer '{layer_name}' not found.")
                return
        # Set the bounding boxes
        poly_name = self.cb_roi_poly.currentText()
        if poly_name not in self.viewer.layers:
            self.model.error(f"Polygons layer '{poly_name}' not found.")
            return
        self._find_areas()
        for color, collection in self.image_areas.items():
            for poly in collection['polygons']:
                self.model.add_region(color, poly)
        for color, collection in self.image_areas.items():
            if collection['starter'] is not None:
                self.model.set_starting_hint(color, collection['starter'])
        self.model.set_calibration(
            self.viewer.layers[seg_layer_name].scale,
            str(self.viewer.layers[seg_layer_name].units[0])
        )
        # If a color (filament) doesn't have a starting point, abort
        for color, collection in self.image_areas.items():
            if collection['starter'] is None:
                self.model.error(f"No starting point defined for color {color}.")
                return
        
        # Run the export in a separate thread
        self.qt_worker = QtExportCrops(self.model)
        self.qt_thread = None
        self.qt_thread = QThread()
        self.qt_worker.moveToThread(self.qt_thread)
        self.qt_thread.started.connect(self.qt_worker.run)
        self.qt_worker.finished.connect(self.qt_thread.quit)
        self.qt_worker.finished.connect(self.qt_worker.deleteLater)
        self.qt_thread.finished.connect(self._finished_export_crops)
        self.qt_thread.start()
        self.set_activated_ui(False)

    def _finished_export_crops(self):
        print("Finished exporting crops.")
        self.set_activated_ui(True)

    def _create_plots(self):
        self.qt_worker = QtPlot(self.model)
        self.qt_thread = None
        self.qt_thread = QThread()
        self.qt_worker.moveToThread(self.qt_thread)
        self.qt_thread.started.connect(self.qt_worker.run)
        self.qt_worker.finished.connect(self.qt_thread.quit)
        self.qt_worker.finished.connect(self.qt_worker.deleteLater)
        self.qt_thread.finished.connect(self._finished_create_plots)
        self.qt_thread.start()
        self.set_activated_ui(False)

    def _finished_create_plots(self):
        print("Finished creating plots.")
        self.set_activated_ui(True)

    def _show_rank(self):
        texts = []
        colors = []
        for color, collection in self.model.chunks.items():
            for bbox in collection['bboxes']:
                r = self.model.get_rank(color, bbox)
                txt = f"R{r}" if r is not None else "?"
                texts.append(txt)
                colors.append(utils.str_to_clr(color))
        area_name = self.cb_roi_poly.currentText()
        text_parameters = {
            'string': texts,
            'size': 14,
            'color': colors,
            'anchor': 'upper_left'
        }
        if area_name in self.viewer.layers:
            layer = self.viewer.layers[area_name]
            layer.text = text_parameters

    def _find_starters(self):
        ln_starting_pts = self.cb_roi_start.currentText()
        l_starting = None
        if ln_starting_pts in self.viewer.layers:
            l_starting = self.viewer.layers[ln_starting_pts]
        if l_starting is None:
            return
        for point in l_starting.data:
            z, y, x = point
            pt = Point(x, y)
            for color, collection in self.image_areas.items():
                for poly in collection['polygons']:
                    polygon = utils.as_polygon(poly)
                    if polygon.contains(pt):
                        collection['starter'] = (z, y, x)
                        break

    def _find_areas(self):
        ln_polys = self.cb_roi_poly.currentText()
        l_polys  = None
        if ln_polys in self.viewer.layers:
            l_polys = self.viewer.layers[ln_polys]
        if l_polys is None:
            return
        self.image_areas = {}
        for index, (color, coords) in enumerate(zip(l_polys.edge_color, l_polys.data)):
            clr_key = clr_to_str(color)
            collection = self.image_areas.setdefault(clr_key, {'polygons': [], 'starter': None, 'indices': []})
            collection['polygons'].append(coords)
            collection['indices'].append(index)
        self._find_starters()

    def _add_measure_row(self, preset_text=None, preset_choice=None):
        """
        Adds a new measures-row: "Cx" | [layers + NEUTRAL] | QLineEdit("---")
        preset_text: optional predefined text for QLineEdit
        preset_choice: optional predefined selection for combobox
        """
        idx = len(self.measure_rows) + 1
        container = QWidget()
        hl = QHBoxLayout(container)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(8)

        lbl = QLabel(f"C{idx}")
        combo = QComboBox()
        self.channel_pools.add(combo)
        # fill later after we have the widget in the tree
        edit = QLineEdit(preset_text or "---")

        hl.addWidget(lbl)
        hl.addWidget(combo, 1)
        hl.addWidget(edit, 1)

        self.meas_v.addWidget(container)

        row_info = {'label': lbl, 'combo': combo, 'edit': edit, 'container': container}
        self.measure_rows.append(row_info)

        # populate choices for this new combobox
        self._populate_layer_combo(combo, neutral=NEUTRAL)

        # connect behavior
        combo.currentIndexChanged.connect(lambda _i, r=row_info: self._on_measure_combo_changed(r))

        # optional preset selection
        if preset_choice is not None:
            self._set_combo_safely(combo, preset_choice)

        return row_info

    def _remove_measure_row(self, row_info):
        """Removes a specific row (widgets + layout) and renumbers remaining."""
        if len(self.measure_rows) <= 1:
            # always keep at least one row
            row_info['combo'].setCurrentText(NEUTRAL)
            row_info['edit'].setText(row_info['label'].text())
            return

        container = row_info['container']
        self.meas_v.removeWidget(container)
        container.setParent(None)
        container.deleteLater()
        self.measure_rows.remove(row_info)
        self._renumber_measure_rows()

    def _renumber_measure_rows(self):
        """Renumber labels and default text (only if edit still matches previous label)."""
        for i, r in enumerate(self.measure_rows, start=1):
            old_label = r['label'].text()
            new_label = f"C{i}"
            # update label
            r['label'].setText(new_label)
            # if user didn’t customize the text (still equals old label), keep it in sync
            if r['edit'].text() == old_label:
                r['edit'].setText(new_label)

    def _on_measure_combo_changed(self, row_info):
        sel = row_info['combo'].currentText()

        if sel == NEUTRAL:
            # remove this row
            self._remove_measure_row(row_info)
        else:
            # when selecting a valid layer, ensure there's an empty row at the end
            # (only if this row is currently the last with content)
            if self._is_last_row(row_info):
                self._add_measure_row()
        # After any change, re-sync layer lists (in case layers changed)
        # and renumber rows
        self.refresh_layer_names()

    def _is_last_row(self, row_info):
        """Return True if row_info is the last row or last non-empty row."""
        return self.measure_rows and (row_info is self.measure_rows[-1])

    # ---------------- LAYER LIST HANDLING ----------------

    def _get_layer_names(self):
        try:
            return [ly.name for ly in self.viewer.layers]
        except Exception:
            return []

    def _populate_layer_combo(self, combo: QComboBox, neutral=NEUTRAL):
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(neutral)
        for name in self._get_layer_names():
            combo.addItem(name)
        # restore selection if still available
        self._set_combo_safely(combo, current)
        combo.blockSignals(False)

    def _set_combo_safely(self, combo: QComboBox, text: str):
        idx = combo.findText(text)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        else:
            # fall back to neutral
            combo.setCurrentIndex(0)

    def refresh_layer_names(self):
        """Call this to refresh all comboboxes with current viewer layers."""
        for combo in self.channel_pools:
            self._populate_layer_combo(combo, neutral=NEUTRAL)

    # ---------------- SETTINGS ----------------

    def _pick_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select existing folder")
        if not path:
            return
        self.selected_folder = path
        self.model = BacteriaDensityWorker()
        self.model.set_working_dir(path)
        self.model.recover()
        self.recover_ui_from_model()

    # ---------------- PUBLIC ACCESSORS ----------------

    def recover_images_ui(self):
        # Recover the segmentation channel
        seg_name, seg_data = self.model.get_segmentation_channel()
        if seg_name is None or seg_data is None:
            return False
        self.viewer.add_image(seg_data, name=seg_name)
        self.cb_seg_channel.setCurrentText(seg_name)
        # Recover the measurement channels
        for name, data in self.model.get_measurement_channels().items():
            self.viewer.add_image(data, name=name)
            row = self.measure_rows[-1]
            row['combo'].setCurrentText(name)
            row['edit'].setText(name)
        return True
    
    def recover_calibration_ui(self):
        for l in self.viewer.layers:
            l.scale = np.array(self.model.calibration)
        self.viewer.scale_bar.unit = self.model.unit
        return True
    
    def recover_regions_ui(self):
        d, _, _ = self.model.get_segmentation_channel()[1].shape
        for str_color, collection in self.model.chunks.items():
            color = utils.str_to_clr(str_color)
            polygons = []
            for bbox in collection['bboxes']:
                (minx, miny, maxx, maxy) = bbox
                poly = np.array([
                    [d//2, minx, miny],
                    [d//2, minx, maxy],
                    [d//2, maxx, maxy],
                    [d//2, maxx, miny]
                ], dtype=np.float32)
                polygons.append(poly)
            if not polygons:
                return False
            colors = np.array([color for _ in polygons])
            self.viewer.add_shapes(polygons, shape_type='polygon', edge_color=colors, face_color='transparent', edge_width=20, name="Areas")
            if collection['start'] is not None:
                self.viewer.add_points(np.array([collection['start']]), name="Hint Points", size=20, face_color="cyan")
        return True
    
    def recover_mask(self):
        if self.model.are_skeletons_generated():
            self._finished_make_masks()
            return True
        return False
    
    def recover_medial_path(self):
        if self.model.are_medial_paths_generated():
            self._finished_make_medial_path()
            return True
        return False
    
    def recover_measures(self):
        if self.model.are_measures_generated():
            self._finished_make_measures()
            return True
        return False
    
    def recover_ui_from_model(self):
        self.set_activated_ui(False)
        if not self.recover_images_ui():
            self.set_activated_ui(True)
            return
        if not self.recover_regions_ui():
            self.set_activated_ui(True)
            return
        if not self.recover_mask():
            self.set_activated_ui(True)
            return
        if not self.recover_medial_path():
            self.set_activated_ui(True)
            return
        if not self.recover_measures():
            self.set_activated_ui(True)
            return
        self.recover_calibration_ui()
        self.set_activated_ui(True)

    def get_segmentation_channel(self):
        val = self.cb_seg_channel.currentText()
        return None if val == NEUTRAL else val

    def get_measures_config(self):
        """
        Returns a list of dicts for rows where a layer is selected:
        [{'name': 'C1', 'layer': 'Layer A', 'alias': 'C1'}, ...]
        """
        out = []
        for r in self.measure_rows:
            layer = r['combo'].currentText()
            if layer and layer != NEUTRAL:
                out.append({
                    'name': r['label'].text(),
                    'layer': layer,
                    'alias': r['edit'].text().strip()
                })
        return out

    def get_rois(self):
        start = self.cb_roi_start.currentText()
        bbox  = self.cb_roi_poly.currentText()
        return {
            'starting_points': None if start == NEUTRAL else start,
            'bounding_boxes': None if bbox  == NEUTRAL else bbox
        }

    def get_settings(self):
        return {
            'pixel_size_um': float(self.sb_pixel.value()),
            'folder': self.selected_folder
        }


def launch_test_procedure():
    import tifffile as tiff

    viewer = napari.Viewer()
    widget = BacteriaDensityWidget(viewer=viewer)
    viewer.window.add_dock_widget(widget)

    print("--- Workflow: Small data ---")

    dapi_path   = "/home/clement/Documents/projects/2119-bacteria-density/small-data/nuclei.tif"
    bactos_path = "/home/clement/Documents/projects/2119-bacteria-density/small-data/bactos.tif"
    dapi        = tiff.imread(dapi_path)
    bactos      = tiff.imread(bactos_path)
    layers      = []

    l1 = viewer.add_image(bactos, name="Bactos")
    l1.contrast_limits = (np.min(bactos), np.max(bactos))
    layers.append(l1)

    l2 = viewer.add_image(dapi, name="Nuclei")
    l2.contrast_limits = (np.min(dapi), np.max(dapi))
    layers.append(l2)

    polygons = [
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
        ]),
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
        ]),
        np.array([
            [   8.      , -109.290596, -296.20166 ],
            [   8.      , -109.290596, 1187.27    ],
            [   8.      ,  689.93353 , 1187.27    ],
            [   8.      ,  689.93353 , -296.20166 ]
        ])
    ]

    colors = np.array([
        [1., 0., 0., 1.],
        [1., 1., 0., 1.],
        [1., 0., 1., 1.]
    ])

    hint_points = np.array([
        [8.0, 1220.98960248, -6.11470609],
        [8.0, 748.9657104  , 1817.38811913],
        [8.0, 426.00199476 , -60.77010412]
    ])

    l3 = viewer.add_shapes(polygons, shape_type='polygon', edge_color=colors, face_color='transparent', edge_width=20, name="Areas")
    layers.append(l3)
    l4 = viewer.add_points(hint_points, name="Hint Points", size=20, face_color="cyan")
    layers.append(l4)

    # Set the values in the GUI
    widget.cb_seg_channel.setCurrentText("Nuclei")
    # First row: Bactos
    widget.measure_rows[0]['combo'].setCurrentText("Bactos")
    widget.measure_rows[0]['edit'].setText("RFP")
    # ROIs
    widget.cb_roi_start.setCurrentText("Hint Points")
    widget.cb_roi_poly.setCurrentText("Areas")
    # Output folder
    widget.selected_folder = "/home/clement/Documents/projects/2119-bacteria-density/small-data/output"

    for l in layers:
        l.scale = (0.152, 0.3250244, 0.3250244)
        l.units = ( "µm",      "µm",      "µm")

    napari.run()

if __name__ == "__main__":
    launch_test_procedure()