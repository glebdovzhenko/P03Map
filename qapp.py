from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import pyqtSignal

import pandas as pd
import numpy as np
import os


class P03MapApplication(QApplication):
    updatePlot = pyqtSignal()

    def __init__(self, *args, **kwargs):
        QApplication.__init__(self, *args, **kwargs)

        self.scan_mot_x = 'cube1_x'
        self.scan_mot_y = 'cube1_y'

        self.data = pd.DataFrame({
            'spectra_q': [],
            'spectra_I': [],
            'mot_x': [],
            'mot_y': [],
            'ioni': [],
            'gauss': [],
            'poly': []
        })

        self.mean_q = np.array([])
        self.mean_I = np.array([])

        self.gauss_estimate_params = None
        self.poly_estimate_params = None
        self.estimate_xlim = None

        self.cmaps = dict((name.replace('.csv', ''), np.loadtxt(os.path.join('cmaps', name)))
                          for name in filter(lambda arg: '.csv' in arg, os.listdir(os.path.join('cmaps'))))

    def apply_cmap(self, vals, cmap :str, log_scale=False):
        if cmap not in self.cmaps:
            raise ValueError('Colormap %s not found' % cmap)

        cmap = self.cmaps[cmap]
        colors = (vals - np.min(vals)) / (np.max(vals) - np.min(vals))

        if log_scale:
            colors = np.log(1. + colors)
            colors = (colors - np.min(colors)) / (np.max(colors) - np.min(colors))

        return np.apply_along_axis(lambda a: np.interp(colors, np.linspace(0., 1., cmap.shape[0]), a), 0, cmap)
