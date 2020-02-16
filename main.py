from sys import argv
import time
#from PyQt5.QtCore import *
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
import PyQt5.QtGui
from PyQt5.uic import loadUi
from PyQt5.uic import loadUiType
from PyQt5 import uic


form_class, base_class = loadUiType('main.ui')
class AppWindow(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.widgetList = []    
        self.show()
    def closeEvent(self, event):
        event.accept()

app = 0
app = QApplication(argv)
w = AppWindow()
w.show()

def input():
    filename = QFileDialog.getOpenFileName(w, 'Open data file', '.', 'Data file (*.txt *.dat)')[0]
    w.line_input.setText(filename)


def spanish_inquisition():
    file = w.line_input.text()
    dopusk = w.doubleSpinBox.value()
    print(file)
    print(dopusk)
 
#w.clearButton.clicked.connect(clear_function) #після кліка на батон збирає інфу з виставлених параметрівq
w.select_input.clicked.connect(input)
w.CalculateButton.clicked.connect(spanish_inquisition)
app.exec_()

#w.ui.pushButton.clicked.connect(w.safeExit)