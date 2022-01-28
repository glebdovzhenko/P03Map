from PyQt5.QtWidgets import QWidget, QPushButton, QGridLayout, QTabWidget, QProgressBar, QLabel, QComboBox
from PyQt5.QtGui import QVector3D
from PyQt5.QtCore import Qt
from pyqtgraph import opengl as gl
import pyqtgraph as pg
import numpy as np
from scipy.interpolate import griddata

from qapp import P03MapApplication
from thread_workers import RefinementThread
from utils import roonizi_estimate


class P03MapCentralWidget(QTabWidget):
    def __init__(self, *args, **kwargs):
        QTabWidget.__init__(self, *args, **kwargs)
        self.qapp = P03MapApplication.instance()

        self.t1 = P03MapTab1(parent=self)
        self.addTab(self.t1, 'Mean spectra')
        self.t2 = P03MapTab2(parent=self)
        self.addTab(self.t2, 'Map')


class P03Map3DPlot(gl.GLViewWidget):
    def __init__(self, *args, **kwargs):
        gl.GLViewWidget.__init__(self, *args, **kwargs)
        self.qapp = P03MapApplication.instance()

        self.text_objs = [
            [-50., -50., 0., "0"],
            [50., -50., 0., "100 μm"],
            [-50., 50., 0., "100 μm"],
            [0., -50., 0., self.qapp.scan_mot_x],
            [-50., 0., 0., self.qapp.scan_mot_y],
        ]

    def paintGL(self, *args, **kwds):
        gl.GLViewWidget.paintGL(self, *args, **kwds)

        self.qglClearColor(Qt.white)
        for to in self.text_objs:
            self.renderText(*to)


class P03MapTab2(QWidget):
    def __init__(self, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)
        self.qapp = P03MapApplication.instance()

        # controls
        self.btn_ref = QPushButton('Refine', parent=self)
        self.btn_stp = QPushButton('Stop', parent=self)
        self.prb = QProgressBar(parent=self)
        self.slbl = QLabel('')
        self.slbl.setMaximumHeight(self.btn_ref.height())
        self.worker = None
        self.cmb_cmap = QComboBox()
        self.lbl_cmap = QLabel('Colormap')
        self.lbl_cmap.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        for k in self.qapp.cmaps.keys():
            self.cmb_cmap.addItem(k)
        self.cmb_zdata = QComboBox()
        self.lbl_zdata = QLabel('Data')
        self.lbl_zdata.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.cmb_zdata.addItems(['Height', 'Width', 'Center'])
        self.cmb_zdata_convert = {
            'Height': 'a',
            'Width': 'sigma',
            'Center': 'c'
        }
        self.cmb_scale = QComboBox()
        self.lbl_scale = QLabel('Scale')
        self.lbl_scale.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.cmb_scale.addItems(['Linear', 'Log', 'Root'])
        self.cmb_ptype = QComboBox()
        self.lbl_ptype = QLabel('Plot')
        self.lbl_ptype.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.cmb_ptype.addItems(['Landscape', 'Flat'])
        self.cmb_interp = QComboBox()
        self.lbl_interp = QLabel('Interpolation')
        self.lbl_interp.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.cmb_interp.addItems(['Points', 'Nearest', 'Linear', 'Cubic'])
        self.void = QLabel('')
        self.btn_view1 = QPushButton('Side view')
        self.btn_view2 = QPushButton('Top view')

        # plot
        self.plot = P03Map3DPlot(parent=self)
        self.plot.setMinimumSize(640, 480)
        self.plot.setCameraPosition(distance=80)
        self.upd_plot()

        # layout
        layout = QGridLayout()
        layout.addWidget(self.btn_ref, 1, 1, 1, 2)
        layout.addWidget(self.btn_stp, 2, 1, 1, 2)
        layout.addWidget(self.prb, 3, 1, 1, 2)
        layout.addWidget(self.slbl, 4, 1, 1, 2)
        layout.addWidget(self.lbl_zdata, 5, 1, 1, 1)
        layout.addWidget(self.cmb_zdata, 5, 2, 1, 1)
        layout.addWidget(self.lbl_scale, 6, 1, 1, 1)
        layout.addWidget(self.cmb_scale, 6, 2, 1, 1)
        layout.addWidget(self.lbl_ptype, 7, 1, 1, 1)
        layout.addWidget(self.cmb_ptype, 7, 2, 1, 1)
        layout.addWidget(self.lbl_interp, 8, 1, 1, 1)
        layout.addWidget(self.cmb_interp, 8, 2, 1, 1)
        layout.addWidget(self.lbl_cmap, 9, 1, 1, 1)
        layout.addWidget(self.cmb_cmap, 9, 2, 1, 1)
        layout.addWidget(self.btn_view1, 10, 1, 1, 1)
        layout.addWidget(self.btn_view2, 10, 2, 1, 1)
        layout.addWidget(self.void, 11, 1, 1, 2)
        layout.addWidget(self.plot, 1, 3, 11, 1)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)
        layout.setColumnStretch(3, 10)
        self.setLayout(layout)

        # slots
        self.btn_ref.clicked.connect(self.on_clicked_btn_ref)
        self.btn_stp.clicked.connect(self.on_clicked_btn_stp)
        self.cmb_cmap.currentIndexChanged.connect(self.on_cic_cmb_cmap)
        self.cmb_zdata.currentIndexChanged.connect(self.on_cic_cmb_zdata)
        self.cmb_scale.currentIndexChanged.connect(self.on_cic_cmb_scale)
        self.cmb_ptype.currentIndexChanged.connect(self.on_cic_cmb_ptype)
        self.cmb_interp.currentIndexChanged.connect(self.on_cic_cmb_interp)
        self.btn_view1.clicked.connect(self.on_clicked_btn_view1)
        self.btn_view2.clicked.connect(self.on_clicked_btn_view2)

    def on_clicked_btn_view1(self):
        self.plot.setCameraPosition(distance=180, elevation=30., azimuth=45.)

    def on_clicked_btn_view2(self):
        self.plot.setCameraPosition(distance=130, elevation=90., azimuth=-90.)

    def on_cic_cmb_interp(self, index):
        self.upd_plot()

    def on_cic_cmb_cmap(self, index):
        self.upd_plot()

    def on_cic_cmb_zdata(self, index):
        self.upd_plot()

    def on_cic_cmb_scale(self, index):
        self.upd_plot()

    def on_cic_cmb_ptype(self, index):
        self.upd_plot()

    def on_clicked_btn_ref(self, *args, **kwargs):
        if self.worker is not None:
            return

        self.worker = RefinementThread()
        self.worker.status.connect(self.status_accept)
        self.worker.start()
        self.btn_ref.setEnabled(False)
        self.prb.setRange(0, self.worker.length)
        self.slbl.setText('')

    def on_clicked_btn_stp(self, *args, **kwargs):
        if self.worker is None:
            return

        self.worker.stop.emit()

    def status_accept(self, status):
        self.prb.setValue(int(status))

        if self.prb.value() == self.worker.length - 1:
            self.prb.setValue(0)
            self.btn_ref.setEnabled(True)
            self.worker = None
            self.slbl.setText(
                'Refinement: %d / %d success' %
                (self.qapp.data.shape[0] - self.qapp.data['gauss'].isna().sum(), self.qapp.data.shape[0])
            )
            self.upd_plot()

    def upd_plot(self):
        for item in self.plot.items:
            item._setView(None)
        self.plot.items = []
        self.plot.update()

        g = gl.GLGridItem(size=QVector3D(100, 100, 1), color=(255, 255, 255, 76))
        self.plot.addItem(g)

        def prepare_zdata(arg, key='a'):
            if not isinstance(arg, dict):
                return np.nan
            elif key not in arg.keys():
                return np.nan
            else:
                return arg[key]

        xdata = self.qapp.data['mot_x'].to_numpy(dtype=float) - 50
        ydata = self.qapp.data['mot_y'].to_numpy(dtype=float) - 50
        zdata = self.qapp.data['gauss'].apply(
            lambda x: prepare_zdata(x, key=self.cmb_zdata_convert[self.cmb_zdata.currentText()])
        ).to_numpy(dtype=float)

        args = ~(np.isnan(xdata) | np.isnan(ydata) | np.isnan(zdata))
        xdata, ydata, zdata = xdata[args], ydata[args], zdata[args]

        if zdata.size == 0:
            return

        if self.cmb_scale.currentText() == 'Linear':
            pass
        elif self.cmb_scale.currentText() == 'Log':
            zdata = np.log(zdata)
        elif self.cmb_scale.currentText() == 'Root':
            zdata = np.sqrt(zdata)
        else:
            return

        zdata = 75. * (zdata - zdata.min()) / (zdata.max() - zdata.min())

        if self.cmb_interp.currentText() == 'Points':
            if self.cmb_ptype.currentText() == 'Landscape':
                self.plot.addItem(
                    gl.GLScatterPlotItem(pos=np.vstack([xdata, ydata, zdata]).T,
                                         color=self.qapp.apply_cmap(zdata, cmap=self.cmb_cmap.currentText()))
                )
            elif self.cmb_ptype.currentText() == 'Flat':
                self.plot.addItem(
                    gl.GLScatterPlotItem(pos=np.vstack([xdata, ydata, 0.1 + 0. * zdata]).T,
                                         color=self.qapp.apply_cmap(zdata, cmap=self.cmb_cmap.currentText()))
                )
            else:
                return

        elif self.cmb_interp.currentText() in ('Nearest', 'Linear', 'Cubic'):
            pos = np.vstack([xdata, ydata, zdata]).T

            grid_xx = np.linspace(xdata.min(), xdata.max(), 200)
            grid_yy = np.linspace(ydata.min(), ydata.max(), 200)
            grid_xy = (
                np.array([grid_xx] * grid_yy.shape[0]), np.array([grid_yy] * grid_xx.shape[0]).T
            )
            grid_zz = griddata(pos[:, :2], pos[:, 2], grid_xy, method=self.cmb_interp.currentText().lower())
            grid_zz = grid_zz.reshape(grid_yy.shape[0], grid_xx.shape[0]).T

            colors = grid_zz.copy()
            colors[np.isnan(colors)] = 0.
            colors = self.qapp.apply_cmap(colors, cmap=self.cmb_cmap.currentText())

            if self.cmb_ptype.currentText() == 'Landscape':
                pass
            elif self.cmb_ptype.currentText() == 'Flat':
                grid_zz = 0.1 + 0. * grid_zz
            else:
                return

            self.plot.addItem(
                gl.GLSurfacePlotItem(x=grid_xx, y=grid_yy, z=grid_zz,
                                     colors=colors,
                                     shader=None, computeNormals=False)
            )
        else:
            return


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
