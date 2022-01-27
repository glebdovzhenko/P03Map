from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QErrorMessage

import os
import numpy as np
import pandas as pd

from qapp import P03MapApplication
from utils import refine_estimate, read_fio


class ThreadWorker(QThread):
    status = pyqtSignal(int)
    stop = pyqtSignal()

    def __init__(self):
        QThread.__init__(self)
        self.qapp = P03MapApplication.instance()
        self.length = 0
        self._stop = False

        self.stop.connect(self.on_stop)

    def on_stop(self):
        self._stop = True


class RefinementThread(ThreadWorker):

    def __init__(self):
        ThreadWorker.__init__(self)
        self.length = self.qapp.data.shape[0]

    def run(self):
        self.qapp.data.loc[:, 'poly'] = None
        self.qapp.data.loc[:, 'gauss'] = None

        if self.length == 0:
            self.status.emit(self.length - 1)

        for ii in range(self.length):
            if self._stop:
                self.status.emit(self.length - 1)
                return

            xdata = self.qapp.data.at[ii, 'spectra_q']
            ydata = self.qapp.data.at[ii, 'spectra_I']
            est_poly = self.qapp.poly_estimate_params
            est_gauss = self.qapp.gauss_estimate_params
            xmin, xmax = self.qapp.estimate_xlim

            xdata, ydata = xdata[(xmin < xdata) & (xdata < xmax)], ydata[(xmin < xdata) & (xdata < xmax)]

            try:
                est_poly, est_gauss = refine_estimate(
                    xdata, ydata, est_poly, est_gauss,
                )
            except (RuntimeError, Warning) as exc:
                print(exc)
            else:
                self.qapp.data.at[ii, 'poly'] = est_poly
                self.qapp.data.at[ii, 'gauss'] = est_gauss

            self.status.emit(ii)


class ImportThread(ThreadWorker):

    def __init__(self, fio_names=[]):
        ThreadWorker.__init__(self)
        self.fio_names = fio_names
        self.length = len(fio_names)

    def run(self):
        if not self.fio_names:
            return

        dir_name = os.path.dirname(self.fio_names[0])
        self.fio_names = list(map(os.path.basename, self.fio_names))

        result = {col: [] for col in self.qapp.data.columns}

        missing_subdirs = []
        for ii, fio_name in enumerate(self.fio_names):
            if self._stop:
                self.status.emit(self.length - 1)
                return

            if fio_name[:-4] not in os.listdir(dir_name):
                missing_subdirs.append(fio_name[:-4])
                continue

            fio_header, fio_data = read_fio(os.path.join(dir_name, fio_name))

            if self.qapp.scan_mot_x not in fio_header.keys() or self.qapp.scan_mot_y not in fio_header.keys():
                continue

            if (self.qapp.scan_mot_x not in fio_data.columns and self.qapp.scan_mot_y not in fio_data.columns) or \
                    (self.qapp.scan_mot_x in fio_data.columns and self.qapp.scan_mot_y in fio_data.columns):
                continue

            if self.qapp.scan_mot_x in fio_data.columns:
                positions_x = fio_data[self.qapp.scan_mot_x].to_list()
                positions_y = [fio_header[self.qapp.scan_mot_y]] * len(positions_x)
            else:
                positions_y = fio_data[self.qapp.scan_mot_y].to_list()
                positions_x = [fio_header[self.qapp.scan_mot_x]] * len(positions_y)

            ioni_values = fio_data['ioni'].to_list()

            spectra_names = sorted(filter(lambda x: x[-4:] == '.dat',
                                          os.listdir(os.path.join(dir_name, fio_name[:-4]))))

            if len(spectra_names) != len(positions_x):
                continue

            spectra_qs, spectra_Is = [], []
            for spectra_name in spectra_names:
                dd = np.loadtxt(
                    os.path.join(os.path.join(dir_name, fio_name[:-4], spectra_name))
                )
                dd = dd.T
                spectra_qs.append((.1 * dd[0]).copy())
                spectra_Is.append(dd[1])

            result['spectra_q'] += spectra_qs
            result['spectra_I'] += spectra_Is
            result['mot_x'] += positions_x
            result['mot_y'] += positions_y
            result['ioni'] += ioni_values
            result['gauss'] += [None] * len(ioni_values)
            result['poly'] += [None] * len(ioni_values)

            self.status.emit(ii)

        if missing_subdirs:
            QErrorMessage().showMessage('Folders not found:\n' + '\n'.join(map(str, missing_subdirs)))

        self.qapp.data = pd.DataFrame(result)
        self.qapp.data = self.qapp.data.astype('object')
        self.qapp.mean_q = np.mean(result['spectra_q'], axis=0)
        self.qapp.mean_I = np.mean(result['spectra_I'], axis=0)
        self.qapp.updatePlot.emit()
