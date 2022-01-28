from PyQt5.QtWidgets import QTabWidget

from qapp import P03MapApplication
from tab1 import P03MapTab1
from tab2 import P03MapTab2


class P03MapCentralWidget(QTabWidget):
    def __init__(self, *args, **kwargs):
        QTabWidget.__init__(self, *args, **kwargs)
        self.qapp = P03MapApplication.instance()

        self.t1 = P03MapTab1(parent=self)
        self.addTab(self.t1, 'Mean spectra')
        self.t2 = P03MapTab2(parent=self)
        self.addTab(self.t2, 'Map')
