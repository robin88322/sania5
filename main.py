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
    #filename = QFileDialog.directory(w, "Choose catalog", ".", QFileDialog.ReadOnly)
    filename = QFileDialog.getExistingDirectory(w, "Select Directory")
    filename = filename + "/"
    w.line_input.setText(filename)


def calculate():
    file = w.line_input.text()
    eta_def = w.doubleSpinBox.value()
    res = []
    res, matrix = givno(file, eta_def)
    w.lineEdit_s1.setText(res[0])
    w.lineEdit_s2.setText(res[1])
    w.lineEdit_s3.setText(res[2])
    w.lineEdit_s4.setText(res[3])
    w.lineEdit_00.setText(matrix[0])
    w.lineEdit_01.setText(matrix[1])
    w.lineEdit_02.setText(matrix[2])
    w.lineEdit_03.setText(matrix[3])
    w.lineEdit_04.setText(matrix[4])
    w.lineEdit_05.setText(matrix[5])
    w.lineEdit_06.setText(matrix[6])
    w.lineEdit_10.setText(matrix[7])
    w.lineEdit_11.setText(matrix[8])
    w.lineEdit_12.setText(matrix[9])
    w.lineEdit_13.setText(matrix[10])
    w.lineEdit_14.setText(matrix[11])
    w.lineEdit_15.setText(matrix[12])
    w.lineEdit_16.setText(matrix[13])
    w.lineEdit_20.setText(matrix[14])
    w.lineEdit_21.setText(matrix[15])
    w.lineEdit_22.setText(matrix[16])
    w.lineEdit_23.setText(matrix[17])
    w.lineEdit_24.setText(matrix[18])
    w.lineEdit_25.setText(matrix[19])
    w.lineEdit_26.setText(matrix[20])
    w.lineEdit_30.setText(matrix[21])
    w.lineEdit_31.setText(matrix[22])
    w.lineEdit_32.setText(matrix[23])
    w.lineEdit_33.setText(matrix[24])
    w.lineEdit_34.setText(matrix[25])
    w.lineEdit_35.setText(matrix[26])
    w.lineEdit_36.setText(matrix[27])
    


def plot():
    eta_def = w.doubleSpinBox.value()
    file = w.line_input.text()
    show_plot(file, eta_def)
    
   # print(file)
 
#w.clearButton.clicked.connect(clear_function) #після кліка на батон збирає інфу з виставлених параметрівq
w.select_input.clicked.connect(input)
w.CalculateButton.clicked.connect(calculate)
w.GrapfButton.clicked.connect(plot)
app.exec_()

#w.ui.pushButton.clicked.connect(w.safeExit)