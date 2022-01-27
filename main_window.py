import pandas as pd
from PyQt5.QtWidgets import QMainWindow, QMenu, QAction, QFileDialog, QErrorMessage, QDesktopWidget

import numpy as np
import os

from qapp import P03MapApplication
from central_widget import P03MapCentralWidget
from thread_workers import ImportThread
from utils import read_fio


class P03MapMainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        # initialization
        QMainWindow.__init__(self, *args, **kwargs)
        self.qapp = P03MapApplication.instance()

        # menu bar
        fileMenu = QMenu("&File", self)
        self._act_open = QAction('Open', self)
        fileMenu.addActions([self._act_open])

        mb = self.menuBar()
        mb.addMenu(fileMenu)

        # menu bar actions
        self._act_open.triggered.connect(self.on_act_open)

        # status bar
        self.sb = self.statusBar()
        self.sb.showMessage('Use File->Open to add .FIO files')

        # central widget
        self.setCentralWidget(P03MapCentralWidget(parent=self))

        # setting window size & position
        screen_size = QDesktopWidget().availableGeometry(self).size()
        self.resize(screen_size * 0.7)
        self.move(int(screen_size.width() * 0.15), int(screen_size.height() * 0.15))

    def on_act_open(self, checked):
        # launching FileDialog
        fio_names, file_type = QFileDialog.getOpenFileNames(
            None, 'Open FIO files', '.', 'FIO Files (*.fio);;All Files'
        )

        # checking that the list is not empty and we got FIO files
        if not fio_names or file_type != 'FIO Files (*.fio)':
            return

        # all files should be in the same directory
        if len(set(map(os.path.dirname, fio_names))) > 1:
            return

        self.worker = ImportThread(fio_names=fio_names)
        self.worker.status.connect(self.status_accept)
        self.worker.start()

        return

    def status_accept(self, status):
        self.sb.showMessage('Opening %d / %d .FIO files' % (status, self.worker.length))

        if status == self.worker.length - 1:
            self.worker = None
            self.sb.showMessage('')