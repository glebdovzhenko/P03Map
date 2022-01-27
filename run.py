import sys

from qapp import P03MapApplication
from main_window import P03MapMainWindow


if __name__ == '__main__':
    app = P03MapApplication(sys.argv)

    mw = P03MapMainWindow()
    mw.show()

    sys.exit(app.exec_())
