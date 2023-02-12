import sys
from PyQt5.QtWidgets import QApplication

from NeuronAnnotator import *


if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = NeuronAnnotator()
    MainWindow.show()
    sys.exit(app.exec_())