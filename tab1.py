from PyQt5.QtWidgets import QWidget, QGridLayout
import pyqtgraph as pg
import numpy as np

from qapp import P03MapApplication
from utils import roonizi_estimate


class P03MapTab1(QWidget):
    def __init__(self, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)
        self.qapp = P03MapApplication.instance()

        # plot
        self.plot = pg.GraphicsLayoutWidget(parent=self)
        self.axes = self.plot.addPlot(title="Mean spectra")
        self.axes.setLabel('bottom', "Q", units="1/Å")
        self.axes.setLabel('left', "I", units="counts")
        self.axes.showGrid(x=True, y=True)
        self.plot_region = pg.LinearRegionItem([0.1, 0.9], movable=True)
        self.gauss_estimate = pg.PlotCurveItem([], [], pen='#ff0000')

        # layout
        layout = QGridLayout()
        layout.addWidget(self.plot, 1, 1, 1, 1)
        self.setLayout(layout)

        # signal slots
        self.qapp.updatePlot.connect(self.on_update_plot)
        self.plot_region.sigRegionChangeFinished.connect(self.on_update_region)

    def on_update_plot(self):
        self.axes.clear()
        self.axes.plot(self.qapp.mean_q, self.qapp.mean_I)

        xmin, xmax = self.axes.viewRange()[0]

        self.plot_region.setRegion([0.5 * (xmin + xmax), 0.6 * (xmin + xmax)])
        self.axes.addItem(self.plot_region)
        self.axes.addItem(self.gauss_estimate)
        self.on_update_region(self.plot_region)

    def on_update_region(self, regionItem):
        lo, hi = regionItem.getRegion()
        xdata, ydata = self.qapp.mean_q, self.qapp.mean_I
        xdata, ydata = xdata[(lo < xdata) & (xdata < hi)], ydata[(lo < xdata) & (xdata < hi)]

        xdata_ = np.linspace(np.min(xdata), np.max(xdata), 5000)
        ydata_ = np.interp(xdata_, xdata, ydata)

        est_p, est_g = roonizi_estimate(ydata_, xdata_, nn=2)
        yest = np.exp(-(xdata - est_g['c']) ** 2 / (2. * est_g['sigma'] ** 2.)) * est_g['a'] + \
               np.polyval(est_p, xdata)

        self.gauss_estimate.setData(xdata, yest)

        self.qapp.gauss_estimate_params = est_g
        self.qapp.poly_estimate_params = est_p
        self.qapp.estimate_xlim = lo, hi
