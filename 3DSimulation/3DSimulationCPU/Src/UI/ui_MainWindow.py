# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.11.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
import vtkmodules.all as vtk
# import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from MPIRF.Config.UIConstantListEn import *

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1200, 650)
        font = QtGui.QFont()
        font.setPointSize(10)
        MainWindow.setFont(font)
        self.WindowTitle=TITLE
        # MainWindow.setStyleSheet("QSpinBox, QDoubleSpinBox{\n"
        #                          "    max-width:100px;\n"
        #                          "    min-height:20px;\n"
        #                          "}")

        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.verticalLayout_13 = QtWidgets.QVBoxLayout(self.centralWidget)
        self.verticalLayout_13.setContentsMargins(2, 2, 2, 2)
        self.verticalLayout_13.setSpacing(6)
        self.verticalLayout_13.setObjectName("verticalLayout_13")

        self.frame_6 = QtWidgets.QFrame(self.centralWidget)
        self.frame_6.setMaximumSize(QtCore.QSize(16777215, 40))
        self.frame_6.setFrameShape(QtWidgets.QFrame.Panel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")

        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.frame_6)
        self.horizontalLayout_7.setContentsMargins(11, 5, 11, 5)
        self.horizontalLayout_7.setSpacing(6)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.toolBtn_GenData = QtWidgets.QToolButton(self.frame_6)

        self.label_7 = QtWidgets.QLabel(self.frame_6)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_7.addWidget(self.label_7)
        self.comboTheme = QtWidgets.QComboBox(self.frame_6)
        self.comboTheme.setMinimumSize(QtCore.QSize(120, 0))
        self.comboTheme.setObjectName("comboTheme")
        self.comboTheme.addItem("")
        self.comboTheme.addItem("")
        self.horizontalLayout_7.addWidget(self.comboTheme)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/images/828.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolBtn_GenData.setIcon(icon)
        self.toolBtn_GenData.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.toolBtn_GenData.setObjectName("toolBtn_GenData")
        self.horizontalLayout_7.addWidget(self.toolBtn_GenData)
        self.toolBtn_Counting = QtWidgets.QToolButton(self.frame_6)

        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/icons/images/216.GIF"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolBtn_Counting.setIcon(icon1)
        self.toolBtn_Counting.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.toolBtn_Counting.setObjectName("toolBtn_Counting")
        self.horizontalLayout_7.addWidget(self.toolBtn_Counting)

        self.toolBtn_Quit = QtWidgets.QToolButton(self.frame_6)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/icons/images/132.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolBtn_Quit.setIcon(icon2)
        self.toolBtn_Quit.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.toolBtn_Quit.setObjectName("toolBtn_Quit")
        self.horizontalLayout_7.addWidget(self.toolBtn_Quit)

        self.verticalLayout_13.addWidget(self.frame_6)
        self.splitter_2 = QtWidgets.QSplitter(self.centralWidget)
        self.splitter_2.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_2.setObjectName("splitter_2")
        self.frameData = QtWidgets.QFrame(self.splitter_2)
        self.frameData.setFrameShape(QtWidgets.QFrame.Panel)
        self.frameData.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frameData.setObjectName("frameData")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.frameData)
        self.verticalLayout_3.setContentsMargins(5, 5, 5, 5)
        self.verticalLayout_3.setSpacing(6)
        self.verticalLayout_3.setObjectName("verticalLayout_3")


        self.splitter = QtWidgets.QSplitter(self.frameData)
        self.splitter.setOrientation(QtCore.Qt.Vertical)
        self.splitter.setObjectName("splitter")

        self.groupBox = QtWidgets.QGroupBox(self.splitter)
        self.groupBox.setMaximumSize(QtCore.QSize(16777215, 150))
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_3.addWidget(self.groupBox,Qt.AlignLeft | Qt.AlignTop)


        self.verticalLayout = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout.setContentsMargins(5, 5, 5, 5)
        self.verticalLayout.setSpacing(6)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame_2 = QtWidgets.QFrame(self.groupBox)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.horizontalLayout_6 = QtWidgets.QGridLayout(self.frame_2)

        self.horizontalLayout_6.setContentsMargins(11, 2, 11, 2)
        self.horizontalLayout_6.setSpacing(6)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")

        self.ParticleTemLabel = QtWidgets.QLabel(self.frame_2)
        self.ParticleTemLabel.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.ParticleTemLabel.setObjectName("ParticleTemLabel")
        self.horizontalLayout_6.addWidget(self.ParticleTemLabel,1,0,1,1)

        self.ParticleTemEdit = QtWidgets.QLineEdit(self.frame_2)
        self.ParticleTemEdit.setMinimumSize(QtCore.QSize(80, 10))
        self.ParticleTemEdit.setObjectName("ParticleTemEdit")
        self.horizontalLayout_6.addWidget(self.ParticleTemEdit,1,1,1,1)

        self.ParticleDiaLabel = QtWidgets.QLabel(self.frame_2)
        self.ParticleDiaLabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.ParticleDiaLabel.setObjectName("ParticleDiaLabel")
        self.horizontalLayout_6.addWidget(self.ParticleDiaLabel,2,0,1,1)

        self.ParticleDiaEdit = QtWidgets.QLineEdit(self.frame_2)
        self.ParticleDiaEdit.setMinimumSize(QtCore.QSize(80, 10))
        self.ParticleDiaEdit.setObjectName("ParticleDiaEdit")
        self.horizontalLayout_6.addWidget(self.ParticleDiaEdit,2,1,1,1)

        self.ParticleSatLabel = QtWidgets.QLabel(self.frame_2)
        self.ParticleSatLabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.ParticleSatLabel.setObjectName("ParticleSatLabel")
        self.horizontalLayout_6.addWidget(self.ParticleSatLabel,3,0,1,1)

        self.ParticleSatEdit = QtWidgets.QLineEdit(self.frame_2)
        self.ParticleSatEdit.setMinimumSize(QtCore.QSize(80, 10))
        self.ParticleSatEdit.setObjectName("ParticleTemEdit")
        self.horizontalLayout_6.addWidget(self.ParticleSatEdit,3,1,1,1)

        self.ParticleConLabel = QtWidgets.QLabel(self.frame_2)
        self.ParticleConLabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.ParticleConLabel.setObjectName("ParticleConLabel")
        self.horizontalLayout_6.addWidget(self.ParticleConLabel,4,0,1,1)

        self.ParticleConEdit = QtWidgets.QLineEdit(self.frame_2)
        self.ParticleConEdit.setMinimumSize(QtCore.QSize(80, 10))
        self.ParticleConEdit.setObjectName("ParticleConEdit")
        self.horizontalLayout_6.addWidget(self.ParticleConEdit,4,1,1,1)

        self.verticalLayout.addWidget(self.frame_2)

        self.groupBox_7 = QtWidgets.QGroupBox(self.splitter)
        self.groupBox_7.setMaximumSize(QtCore.QSize(16777215, 300))
        self.groupBox_7.setObjectName("groupBox_7")
        self.verticalLayout_3.addWidget(self.groupBox_7, Qt.AlignLeft | Qt.AlignTop)

        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.groupBox_7)
        self.verticalLayout_2.setContentsMargins(5, 5, 5, 5)
        self.verticalLayout_2.setSpacing(6)
        self.verticalLayout_2.setObjectName("verticalLayout_2")

        self.groupBox_71 = QtWidgets.QGroupBox(self.splitter)
        self.groupBox_71.setObjectName("groupBox_71")
        self.verticalLayout_21 = QtWidgets.QGridLayout(self.groupBox_71)
        self.verticalLayout_21.setContentsMargins(5, 5, 5, 5)
        self.verticalLayout_21.setSpacing(6)
        self.verticalLayout_21.setObjectName("verticalLayout_21")

        self.HardXLabel = QtWidgets.QLabel(self.frame_2)
        self.HardXLabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.HardXLabel.setObjectName("HardXLabel")
        self.verticalLayout_21.addWidget(self.HardXLabel, 0, 1)

        self.HardYLabel = QtWidgets.QLabel(self.frame_2)
        self.HardYLabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.HardYLabel.setObjectName("HardYLabel")
        self.verticalLayout_21.addWidget(self.HardYLabel, 0, 2)

        self.HardZLabel = QtWidgets.QLabel(self.frame_2)
        self.HardZLabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.HardZLabel.setObjectName("HardZLabel")
        self.verticalLayout_21.addWidget(self.HardZLabel, 0, 3)

        self.HardSelGraLabel = QtWidgets.QLabel(self.frame_2)
        self.HardSelGraLabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.HardSelGraLabel.setObjectName("HardSelGraLabel")
        self.verticalLayout_21.addWidget(self.HardSelGraLabel, 1, 0)

        self.HardSelGraXEdit = QtWidgets.QLineEdit(self.frame_2)
        self.HardSelGraXEdit.setMinimumSize(QtCore.QSize(30, 10))
        self.HardSelGraXEdit.setObjectName("HardSelGraXEdit")
        self.verticalLayout_21.addWidget(self.HardSelGraXEdit, 1, 1)

        self.HardSelGraYEdit = QtWidgets.QLineEdit(self.frame_2)
        self.HardSelGraYEdit.setMinimumSize(QtCore.QSize(30, 10))
        self.HardSelGraYEdit.setObjectName("HardSelGraYEdit")
        self.verticalLayout_21.addWidget(self.HardSelGraYEdit, 1, 2)

        self.HardSelGraZEdit = QtWidgets.QLineEdit(self.frame_2)
        self.HardSelGraZEdit.setMinimumSize(QtCore.QSize(30, 10))
        self.HardSelGraZEdit.setObjectName("HardSelGraZEdit")
        self.verticalLayout_21.addWidget(self.HardSelGraZEdit, 1, 3)

        self.HardDriFreLabel = QtWidgets.QLabel(self.frame_2)
        self.HardDriFreLabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.HardDriFreLabel.setObjectName("HardDriFreLabel")
        self.verticalLayout_21.addWidget(self.HardDriFreLabel, 2, 0)

        self.HardDriFreXEdit = QtWidgets.QLineEdit(self.frame_2)
        self.HardDriFreXEdit.setMinimumSize(QtCore.QSize(30, 10))
        self.HardDriFreXEdit.setObjectName("HardDriFreXEdit")
        self.verticalLayout_21.addWidget(self.HardDriFreXEdit, 2, 1)

        self.HardDriFreYEdit = QtWidgets.QLineEdit(self.frame_2)
        self.HardDriFreYEdit.setMinimumSize(QtCore.QSize(30, 10))
        self.HardDriFreYEdit.setObjectName("HardDriFreYEdit")
        self.verticalLayout_21.addWidget(self.HardDriFreYEdit, 2, 2)

        self.HardDriFreZEdit = QtWidgets.QLineEdit(self.frame_2)
        self.HardDriFreZEdit.setMinimumSize(QtCore.QSize(30, 10))
        self.HardDriFreZEdit.setObjectName("HardDriFreZEdit")
        self.verticalLayout_21.addWidget(self.HardDriFreZEdit, 2, 3)

        self.HardDriAmpLabel = QtWidgets.QLabel(self.frame_2)
        self.HardDriAmpLabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.HardDriAmpLabel.setObjectName("HardDriAmpLabel")
        self.verticalLayout_21.addWidget(self.HardDriAmpLabel, 3, 0)

        self.HardDriAmpXEdit = QtWidgets.QLineEdit(self.frame_2)
        self.HardDriAmpXEdit.setMinimumSize(QtCore.QSize(30, 10))
        self.HardDriAmpXEdit.setObjectName("HardDriAmpXEdit")
        self.verticalLayout_21.addWidget(self.HardDriAmpXEdit, 3, 1)

        self.HardDriAmpYEdit = QtWidgets.QLineEdit(self.frame_2)
        self.HardDriAmpYEdit.setMinimumSize(QtCore.QSize(30, 10))
        self.HardDriAmpYEdit.setObjectName("HardDriAmpYEdit")
        self.verticalLayout_21.addWidget(self.HardDriAmpYEdit, 3, 2)

        self.HardDriAmpZEdit = QtWidgets.QLineEdit(self.frame_2)
        self.HardDriAmpZEdit.setMinimumSize(QtCore.QSize(30, 10))
        self.HardDriAmpZEdit.setObjectName("HardDriAmpZEdit")
        self.verticalLayout_21.addWidget(self.HardDriAmpZEdit, 3, 3)

        self.groupBox_72 = QtWidgets.QGroupBox(self.splitter)
        self.groupBox_72.setObjectName("groupBox_72")
        self.verticalLayout_22 = QtWidgets.QGridLayout(self.groupBox_72)
        self.verticalLayout_22.setContentsMargins(5, 5, 5, 5)
        self.verticalLayout_22.setSpacing(6)
        self.verticalLayout_22.setObjectName("verticalLayout_21")

        self.HardRepTimeLabel = QtWidgets.QLabel(self.frame_2)
        self.HardRepTimeLabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.HardRepTimeLabel.setObjectName("HardRepTimeLabel")
        self.verticalLayout_22.addWidget(self.HardRepTimeLabel, 0, 0)

        self.HardRepTimeEdit = QtWidgets.QLineEdit(self.frame_2)
        self.HardRepTimeEdit.setMinimumSize(QtCore.QSize(30, 10))
        self.HardRepTimeEdit.setObjectName("HardRepTimeEdit")
        self.verticalLayout_22.addWidget(self.HardRepTimeEdit, 0, 1)

        self.HardSamFreLabel = QtWidgets.QLabel(self.frame_2)
        self.HardSamFreLabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.HardSamFreLabel.setObjectName("HardSamFreLabel")
        self.verticalLayout_22.addWidget(self.HardSamFreLabel, 1, 0)

        self.HardSamFreEdit = QtWidgets.QLineEdit(self.frame_2)
        self.HardSamFreEdit.setMinimumSize(QtCore.QSize(30, 10))
        self.HardSamFreEdit.setObjectName("HardSamFreEdit")
        self.verticalLayout_22.addWidget(self.HardSamFreEdit, 1, 1)

        self.verticalLayout_2.addWidget(self.groupBox_71)
        self.verticalLayout_2.addWidget(self.groupBox_72)


        ################################################################################################################
        self.groupBox_8 = QtWidgets.QGroupBox(self.splitter)
        self.groupBox_8.setMaximumSize(QtCore.QSize(16777215, 80))
        self.groupBox_8.setObjectName("groupBox_8")
        self.verticalLayout_3.addWidget(self.groupBox_8, Qt.AlignLeft | Qt.AlignTop)

        self.horizontalLayout_8 = QtWidgets.QGridLayout(self.groupBox_8)
        self.horizontalLayout_8.setContentsMargins(11, 2, 11, 2)
        self.horizontalLayout_8.setSpacing(6)
        self.horizontalLayout_8.setObjectName("horizontalLayout_6")

        self.ComProcLabel = QtWidgets.QLabel(self.frame_2)
        self.ComProcLabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.ComProcLabel.setObjectName("ComProcLabel")
        self.horizontalLayout_8.addWidget(self.ComProcLabel, 0, 0)

        self.comboThemeProc = QtWidgets.QComboBox(self.frame_2)
        self.comboThemeProc.setMinimumSize(QtCore.QSize(120, 0))
        self.comboThemeProc.setObjectName("comboThemeProc")
        self.comboThemeProc.addItem("")
        self.horizontalLayout_8.addWidget(self.comboThemeProc, 0, 1)

        self.ComThdsLabel = QtWidgets.QLabel(self.frame_2)
        self.ComThdsLabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.ComThdsLabel.setObjectName("ComProcLabel")
        self.horizontalLayout_8.addWidget(self.ComThdsLabel, 1, 0)

        self.comboThemeThds = QtWidgets.QComboBox(self.frame_2)
        self.comboThemeThds.setMinimumSize(QtCore.QSize(120, 0))
        self.comboThemeThds.setObjectName("comboThemeThds")
        self.comboThemeThds.addItem("")
        self.comboThemeThds.addItem("")
        self.comboThemeThds.addItem("")
        self.horizontalLayout_8.addWidget(self.comboThemeThds, 1, 1)

        self.verticalLayout_3.addWidget(self.splitter)
        ################################################################################################################


        self.tabWidget = QtWidgets.QTabWidget(self.splitter_2)
        self.tabWidget.setTabPosition(QtWidgets.QTabWidget.North)
        self.tabWidget.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.tabWidget.setDocumentMode(True)
        self.tabWidget.setObjectName("tabWidget")

        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.tab)
        self.verticalLayout_9.setContentsMargins(2, 2, 2, 2)
        self.verticalLayout_9.setSpacing(6)
        self.verticalLayout_9.setObjectName("verticalLayout_9")

        self.widget = QtWidgets.QWidget(self.tab)
        self.widget.setObjectName("widget")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_8.setSpacing(0)
        self.verticalLayout_8.setObjectName("verticalLayout_8")

        self.frame = QtWidgets.QFrame(self.widget)
        self.frame.setFrameShape(QtWidgets.QFrame.Panel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout.setContentsMargins(11, 5, 11, 5)
        self.horizontalLayout.setSpacing(12)
        self.horizontalLayout.setObjectName("horizontalLayout")

        self.ParticleRelLabel = QtWidgets.QLabel(self.frame)
        self.ParticleRelLabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.ParticleRelLabel.setObjectName("ParticleRelLabel")
        self.horizontalLayout.addWidget(self.ParticleRelLabel)

        self.ParticleRelEdit = QtWidgets.QLineEdit(self.frame)
        self.ParticleRelEdit.setMinimumSize(QtCore.QSize(60, 10))
        self.ParticleRelEdit.setObjectName("ParticleRelEdit")
        self.horizontalLayout.addWidget(self.ParticleRelEdit)

        self.ParticleRelchkBox = QtWidgets.QCheckBox(self.frame)
        # self.ParticleRelchkBox.setChecked(True)
        self.ParticleRelchkBox.setObjectName("ParticleRelchkBox")
        self.horizontalLayout.addWidget(self.ParticleRelchkBox)

        self.NoiseLabel = QtWidgets.QLabel(self.frame)
        self.NoiseLabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.NoiseLabel.setObjectName("NoiseLabel")
        self.horizontalLayout.addWidget(self.NoiseLabel)

        self.NoiseEdit = QtWidgets.QLineEdit(self.frame)
        self.NoiseEdit.setMinimumSize(QtCore.QSize(60, 10))
        self.NoiseEdit.setObjectName("NoiseEdit")
        self.horizontalLayout.addWidget(self.NoiseEdit)

        self.NoisechkBox = QtWidgets.QCheckBox(self.frame)
        # self.NoisechkBox.setChecked(True)
        self.NoisechkBox.setObjectName("NoisechkBox")
        self.horizontalLayout.addWidget(self.NoisechkBox)

        self.BackgroundLabel = QtWidgets.QLabel(self.frame)
        self.BackgroundLabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.BackgroundLabel.setObjectName("BackgroundLabel")
        self.horizontalLayout.addWidget(self.BackgroundLabel)

        self.BackgroundEdit = QtWidgets.QLineEdit(self.frame)
        self.BackgroundEdit.setMinimumSize(QtCore.QSize(60, 10))
        self.BackgroundEdit.setObjectName("BackgroundEdit")
        self.horizontalLayout.addWidget(self.BackgroundEdit)

        self.BackgroundchkBox = QtWidgets.QCheckBox(self.frame)
        # self.BackgroundchkBox.setChecked(True)
        self.BackgroundchkBox.setObjectName("BackgroundchkBox")
        self.horizontalLayout.addWidget(self.BackgroundchkBox)

        self.InterferenceBtn = QtWidgets.QPushButton(self.frame)
        self.InterferenceBtn.setObjectName("InterferenceBtn")
        self.horizontalLayout.addWidget(self.InterferenceBtn)

        spacerItem2 = QtWidgets.QSpacerItem(662, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem2)
        self.verticalLayout_8.addWidget(self.frame)

        colors = vtk.vtkNamedColors()
        self.vtkWidget = QVTKRenderWindowInteractor(self.widget)
        self.ren1 = vtk.vtkRenderer()
        self.ren2 = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren1)
        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren2)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()

        self.ren1.SetViewport(0, 0, 0.5, 1)
        self.ren1.SetBackground(colors.GetColor3d('Black'))
        self.ren2.SetViewport(0.5, 0, 1, 1)
        self.ren2.SetBackground(colors.GetColor3d('Black'))
        self.iren.Initialize()

        self.verticalLayout_8.addWidget(self.vtkWidget)
        self.verticalLayout_9.addWidget(self.widget)

        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/icons/images/3.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.tabWidget.addTab(self.tab, icon3, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.verticalLayout_10 = QtWidgets.QVBoxLayout(self.tab_3)
        self.verticalLayout_10.setContentsMargins(2, 2, 2, 2)
        self.verticalLayout_10.setSpacing(6)
        self.verticalLayout_10.setObjectName("verticalLayout_10")

        self.widget_2 = QtWidgets.QWidget(self.tab_3)
        self.widget_2.setObjectName("widget_2")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.widget_2)
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_5.setSpacing(0)
        self.verticalLayout_5.setObjectName("verticalLayout_5")

        self.frame_3 = QtWidgets.QFrame(self.widget_2)
        self.frame_3.setMaximumSize(QtCore.QSize(17777215, 40))
        self.frame_3.setFrameShape(QtWidgets.QFrame.Panel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.frame_3)
        self.horizontalLayout_3.setContentsMargins(11, 5, 11, 5)
        self.horizontalLayout_3.setSpacing(12)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")

        self.comboThemePlane = QtWidgets.QComboBox(self.frame_3)
        self.comboThemePlane.setMinimumSize(QtCore.QSize(120, 0))
        self.comboThemePlane.setObjectName("comboThemePlane")
        self.comboThemePlane.addItem("")
        self.comboThemePlane.addItem("")
        self.comboThemePlane.addItem("")
        self.comboThemePlane.addItem("")
        self.comboThemePlane.addItem("")
        self.comboThemePlane.addItem("")
        self.horizontalLayout_3.addWidget(self.comboThemePlane)

        self.ImageSliceslider = QtWidgets.QSlider(self.frame_3)
        self.ImageSliceslider.setMinimumSize(QtCore.QSize(60, 10))
        self.ImageSliceslider.setMaximum(1)
        self.ImageSliceslider.setProperty("value", 1)
        self.ImageSliceslider.setOrientation(QtCore.Qt.Horizontal)
        self.ImageSliceslider.setTickPosition(QtWidgets.QSlider.NoTicks)
        self.ImageSliceslider.setObjectName("ImageSliceslider")
        self.horizontalLayout_3.addWidget(self.ImageSliceslider)

        self.ImageSlicelabel = QtWidgets.QLabel(self.frame_3)
        self.ImageSlicelabel.setObjectName("ImageSlicelabel")
        self.horizontalLayout_3.addWidget(self.ImageSlicelabel)

        self.btnBuildStackedBar = QtWidgets.QPushButton(self.frame_3)
        self.btnBuildStackedBar.setObjectName("btnBuildStackedBar")
        self.horizontalLayout_3.addWidget(self.btnBuildStackedBar)

        spacerItem3 = QtWidgets.QSpacerItem(536, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem3)
        self.verticalLayout_5.addWidget(self.frame_3)
        self.ProjectionView = QmyFigureCanvas(self.widget_2)
        self.ProjectionView.setObjectName("ProjectionView")
        self.verticalLayout_5.addWidget(self.ProjectionView)
        self.verticalLayout_10.addWidget(self.widget_2)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/icons/images/281.GIF"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.tabWidget.addTab(self.tab_3, icon4, "")
        self.tab_5 = QtWidgets.QWidget()
        self.tab_5.setObjectName("tab_5")
        self.verticalLayout_12 = QtWidgets.QVBoxLayout(self.tab_5)
        self.verticalLayout_12.setContentsMargins(2, 2, 2, 2)
        self.verticalLayout_12.setSpacing(6)
        self.verticalLayout_12.setObjectName("verticalLayout_12")
        self.widget_4 = QtWidgets.QWidget(self.tab_5)
        self.widget_4.setObjectName("widget_4")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.widget_4)
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_6.setSpacing(0)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.frame_5 = QtWidgets.QFrame(self.widget_4)
        self.frame_5.setFrameShape(QtWidgets.QFrame.Panel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.frame_5)
        self.horizontalLayout_4.setContentsMargins(11, 5, 11, 5)
        self.horizontalLayout_4.setSpacing(12)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.btnPercentBar = QtWidgets.QPushButton(self.frame_5)
        self.btnPercentBar.setObjectName("btnPercentBar")
        self.horizontalLayout_4.addWidget(self.btnPercentBar)
        self.btnPercentBarH = QtWidgets.QPushButton(self.frame_5)
        self.btnPercentBarH.setObjectName("btnPercentBarH")
        self.horizontalLayout_4.addWidget(self.btnPercentBarH)
        spacerItem4 = QtWidgets.QSpacerItem(523, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem4)
        self.verticalLayout_6.addWidget(self.frame_5)
        self.chartViewPercentBar = QmyFigureCanvas(self.widget_4)
        self.chartViewPercentBar.setObjectName("chartViewPercentBar")
        self.verticalLayout_6.addWidget(self.chartViewPercentBar)
        self.verticalLayout_12.addWidget(self.widget_4)
        self.verticalLayout_13.addWidget(self.splitter_2)
        MainWindow.setCentralWidget(self.centralWidget)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        self.toolBtn_Quit.clicked.connect(MainWindow.close)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):

        _translate = QtCore.QCoreApplication.translate

        self.toolBtn_GenData.setToolTip(_translate("MainWindow", SIMUBTN))
        self.toolBtn_GenData.setText(_translate("MainWindow", SIMUBTN))
        self.toolBtn_Counting.setToolTip(_translate("MainWindow", EXPORTDATA))
        self.toolBtn_Counting.setText(_translate("MainWindow", EXPORTDATA))
        self.label_7.setText(_translate("MainWindow", PHANSHAPE))
        self.comboTheme.setItemText(0, _translate("MainWindow", "E Shape"))
        self.comboTheme.setItemText(1, _translate("MainWindow", "P Shape"))
        self.toolBtn_Quit.setText(_translate("MainWindow", "EXIT"))

        self.groupBox.setTitle(_translate("MainWindow", PARPARA))
        self.ParticleTemLabel.setText(_translate("MainWindow", PARTTEMPE+"(℃)"))
        self.ParticleTemEdit.setText("19.85")
        self.ParticleDiaLabel.setText(_translate("MainWindow", PARTDIA+"(nm)"))
        self.ParticleDiaEdit.setText("27")
        self.ParticleSatLabel.setText(_translate("MainWindow", PARTSATMAG+"(T)"))
        self.ParticleSatEdit.setText("1")
        self.ParticleConLabel.setText(_translate("MainWindow", PARTCON+"(mmol/L)"))
        self.ParticleConEdit.setText("100")

        self.groupBox_7.setTitle(_translate("MainWindow", SCANNERPARA))
        self.HardXLabel.setText(_translate("MainWindow", "X"))
        self.HardYLabel.setText(_translate("MainWindow", "Y"))
        self.HardZLabel.setText(_translate("MainWindow", "Z"))
        self.HardSelGraLabel.setText(_translate("MainWindow", SELECTFIELDGRAD+"(T/m)"))
        self.HardSelGraXEdit.setText("2.75")
        self.HardSelGraYEdit.setText("2.75")
        self.HardSelGraZEdit.setText("5.5")
        self.HardDriFreLabel.setText(_translate("MainWindow", DRIVEFIELDFREQ+"(KHz)"))
        self.HardDriFreXEdit.setText("24.51")
        self.HardDriFreYEdit.setText("26.04")
        self.HardDriFreZEdit.setText("25.25")
        self.HardDriAmpLabel.setText(_translate("MainWindow", DRIVEFIELDAMP+"(mT)"))
        self.HardDriAmpXEdit.setText("6")
        self.HardDriAmpYEdit.setText("6")
        self.HardDriAmpZEdit.setText("6")
        self.HardRepTimeLabel.setText(_translate("MainWindow", REPEATTIME+"(ms)"))
        self.HardRepTimeEdit.setText("21.54")
        self.HardSamFreLabel.setText(_translate("MainWindow", SAMPLEREQ+"(MHz)"))
        self.HardSamFreEdit.setText("0.25")

        self.ParticleRelLabel.setText(_translate("MainWindow", PARTICLEREL+"(us)"))
        self.ParticleRelEdit.setText("3")
        self.ParticleRelchkBox.setText(_translate("MainWindow", ADDREL))
        self.ParticleRelchkBox.setCheckState(Qt.Unchecked)
        self.NoiseLabel.setText(_translate("MainWindow", NOISE+"(dB)"))
        self.NoiseEdit.setText("25")
        self.NoisechkBox.setText(_translate("MainWindow", ADDNOISE))
        self.NoisechkBox.setCheckState(Qt.Unchecked)
        self.BackgroundLabel.setText(_translate("MainWindow", BACKGROUNDINTER+"(dB)"))
        self.BackgroundEdit.setText("5")
        self.BackgroundchkBox.setText(_translate("MainWindow", ADDBACKINTER))
        self.BackgroundchkBox.setCheckState(Qt.Unchecked)
        self.InterferenceBtn.setText(_translate("MainWindow", ADDINTER))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", TDVISUAL))

        self.groupBox_8.setTitle(_translate("MainWindow", PERPARA))
        self.ComProcLabel.setText(_translate("MainWindow", PROTYPE))
        self.comboThemeProc.setItemText(0, _translate("MainWindow", "CPU"))
        self.ComThdsLabel.setText(_translate("MainWindow", MAXTHDNUM))
        self.comboThemeThds.setItemText(0, _translate("MainWindow", "4"))
        self.comboThemeThds.setItemText(1, _translate("MainWindow", "8"))
        self.comboThemeThds.setItemText(2, _translate("MainWindow", "16"))

        self.comboThemePlane.setItemText(0, _translate("MainWindow", XYMAX))
        self.comboThemePlane.setItemText(1, _translate("MainWindow", XYSLICE))
        self.comboThemePlane.setItemText(2, _translate("MainWindow", XZMAX))
        self.comboThemePlane.setItemText(3, _translate("MainWindow", XZSLICE))
        self.comboThemePlane.setItemText(4, _translate("MainWindow", YZMAX))
        self.comboThemePlane.setItemText(5, _translate("MainWindow", YZSLICE))
        self.ImageSliceslider.setVisible(False)
        self.ImageSlicelabel.setVisible(False)
        self.ImageSlicelabel.setText(_translate("MainWindow", "1"))
        self.btnBuildStackedBar.setText(_translate("MainWindow", PROJECTION))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", TDPROJECTION))

from UI.myFigureCanvas import QmyFigureCanvas
import UI.res_rc
