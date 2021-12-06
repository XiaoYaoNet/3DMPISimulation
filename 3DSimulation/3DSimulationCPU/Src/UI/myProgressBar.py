from PyQt5.QtWidgets import QApplication, QDialog, QProgressBar
from PyQt5.QtCore import *
import sys


class myProgressBar(QDialog):
    def __init__(self, parent=None):
        super(myProgressBar, self).__init__(parent)

        self.resize(500, 32)

        self.progressBar = QProgressBar(self)
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(100)
        self.progressBar.setValue(0)
        self.progressBar.setGeometry(QRect(1, 3, 499, 28))
        self.setWindowFlags(Qt.WindowMaximizeButtonHint | Qt.MSWindowsFixedSizeDialogHint)
        self.show()

    def setValue(self, task_number, total_task_number, str_value):
        task_number=task_number
        if total_task_number == 0:
            self.setWindowTitle(self.tr(str_value))
            self.progressBar.setValue(task_number)
        else:
            label = str_value
            self.setWindowTitle(self.tr(label))
            p=int((task_number/total_task_number)*100)
            self.progressBar.setValue(p)
        QApplication.processEvents()