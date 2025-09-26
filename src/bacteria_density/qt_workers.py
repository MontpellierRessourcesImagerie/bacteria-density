from qtpy.QtCore import QObject
from PyQt5.QtCore import pyqtSignal, pyqtSlot
import numpy as np

class QtExportCrops(QObject):
    
    finished = pyqtSignal()

    def __init__(self, model):
        super().__init__()
        self.model = model

    def run(self):
        self.model.export_regions()
        self.finished.emit()

class QtMakeMasks(QObject):
    
    finished = pyqtSignal()

    def __init__(self, model):
        super().__init__()
        self.model = model

    def run(self):
        self.model.make_mask_and_skeleton()
        self.finished.emit()

class QtMedialAxis(QObject):
    
    finished = pyqtSignal()

    def __init__(self, model):
        super().__init__()
        self.model = model

    def run(self):
        self.model.make_medial_path()
        self.finished.emit()

class QtMeasure(QObject):
    
    finished = pyqtSignal()

    def __init__(self, model):
        super().__init__()
        self.model = model

    def run(self):
        self.model.make_measures()
        self.finished.emit()

class QtPlot(QObject):
    
    finished = pyqtSignal()

    def __init__(self, model):
        super().__init__()
        self.model = model

    def run(self):
        self.model.make_plots()
        self.finished.emit()