from sys import argv
import time
#from PyQt5.QtCore import *
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
import PyQt5.QtGui
from PyQt5.uic import loadUi
from PyQt5.uic import loadUiType
from PyQt5 import uic
from logic import *


form_class, base_class = loadUiType('data/main.ui')
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
    filename = QFileDialog.getOpenFileName(w, 'Open data file', '.', 'Data file (*.txt *.csv)')[0]
    w.line_input.setText(filename)


def calculate():
    file = w.line_input.text()
    eta_def = w.doubleSpinBox.value()
    res = []
    res = givno(eta_def)
    w.lineEdit_s1.setText(res[0])
    w.lineEdit_s2.setText(res[1])
    w.lineEdit_s3.setText(res[2])
    w.lineEdit_s4.setText(res[3])

def plot():
    eta_def = w.doubleSpinBox.value()
    show_plot(eta_def)
    
   # print(file)
 
#w.clearButton.clicked.connect(clear_function) #після кліка на батон збирає інфу з виставлених параметрівq
w.select_input.clicked.connect(input)
w.CalculateButton.clicked.connect(calculate)
w.GrapfButton.clicked.connect(plot)
app.exec_()

#w.ui.pushButton.clicked.connect(w.safeExit)