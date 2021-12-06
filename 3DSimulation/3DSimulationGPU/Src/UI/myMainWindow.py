# -*- coding: utf-8 -*-
import math
import sys, random

from PyQt5.QtWidgets import  QApplication, QMainWindow, QFileDialog

from PyQt5.QtCore import  Qt,pyqtSlot

from PyQt5.QtGui import QStandardItemModel,QStandardItem, QPainter, QPen

from PyQt5.QtChart import *

import vtkmodules.all as vtk

import MPIRF.Config.ConstantList
from UI.ui_MainWindow import  Ui_MainWindow

from Model.Control import *

from skimage.metrics import structural_similarity as ssim
from MPIRF.Config.UIConstantListEn import TDVISUALREND

import json
import ast

from MPIRF.Config.ConstantList import SET_VALUE

class QmyMainWindow(QMainWindow):

   def __init__(self, parent=None):
      super().__init__(parent)
      self.ui=Ui_MainWindow()
      self.ui.setupUi(self)

      self.setWindowTitle(self.ui.WindowTitle)

      self.setStyleSheet("QTreeWidget, QTableView{"
                        "alternate-background-color:rgb(170, 241, 190)}")

      self.Message=None

      self.ImgData=None
      self.OrgImgData=None
      self.renInit()
      self.ui.ImageSliceslider.valueChanged.connect(self.do_valueChanged)

   def do_valueChanged(self, value):
      self.ui.ImageSlicelabel.setText("Layer "+str(value+1)+"")
      self.ui.ProjectionView.figure.clear()
      ax1 = self.ui.ProjectionView.figure.add_subplot(1, 1, 1, label="points")
      l = np.shape(self.ImgData)[0]
      r = np.shape(self.ImgData)[1]
      o = np.shape(self.ImgData)[2]
      index = self.ui.comboThemePlane.currentIndex()
      if index == 0:
         Ixy = np.zeros((l, r))
         for i in range(l):
            for j in range(r):
               Ixy[i, j] = max(self.ImgData[i, j, :])
         Ixy = Ixy / np.max(Ixy)
         ax1.imshow(Ixy, cmap=plt.get_cmap('binary'))
         ax1.set_title("X-Y")
         ax1.axis("off")
      elif index == 1:
         Ixy = np.zeros((l, r))
         Ixy[:, :] = self.ImgData[:, :, int(self.ui.ImageSliceslider.value())]
         Ixy = Ixy / np.max(Ixy)
         ax1.imshow(Ixy, cmap=plt.get_cmap('binary'))
         ax1.set_title("X-Y")
         ax1.axis("off")
      elif index == 2:
         Ixz = np.zeros((l, o))
         for i in range(l):
            for j in range(o):
               Ixz[i, j] = max(self.ImgData[i, :, j])
         Ixz = Ixz / np.max(Ixz)
         ax1.imshow(Ixz, cmap=plt.get_cmap('binary'))
         ax1.set_title("X-Z")
         ax1.axis("off")
      elif index == 3:
         Ixz = np.zeros((l, o))
         Ixz[:, :] = self.ImgData[:, int(self.ui.ImageSliceslider.value()), :]
         Ixz = Ixz / np.max(Ixz)
         ax1.imshow(Ixz, cmap=plt.get_cmap('binary'))
         ax1.set_title("X-Z")
         ax1.axis("off")
      elif index == 4:
         Iyz = np.zeros((r, o))
         for i in range(r):
            for j in range(o):
               Iyz[i, j] = max(self.ImgData[:, i, j])
         Iyz = Iyz / np.max(Iyz)
         ax1.imshow(Iyz, cmap=plt.get_cmap('binary'))
         ax1.set_title("Y-Z")
         ax1.axis("off")
      elif index == 5:
         Iyz = np.zeros((r, o))
         Iyz[:, :] = self.ImgData[int(self.ui.ImageSliceslider.value()), :, :]
         Iyz = Iyz / np.max(Iyz)
         ax1.imshow(Iyz, cmap=plt.get_cmap('binary'))
         ax1.set_title("Y-Z")
         ax1.axis("off")

      self.ui.ProjectionView.redraw()

   def renInit(self):
      self.OrgImgData=get_OriImgData()
      self.ren1(self.OrgImgData)
      self.ui.iren.Initialize()

   def ren1(self, OrgImgData):

      self.ui.vtkWidget.GetRenderWindow().RemoveRenderer(self.ui.ren1)
      self.ui.ren1 = vtk.vtkRenderer()
      self.ui.vtkWidget.GetRenderWindow().AddRenderer(self.ui.ren1)
      self.ui.ren1.SetViewport(0.5, 0, 1, 1)
      self.ui.ren1.SetBackground(vtk.vtkNamedColors().GetColor3d('Black'))

      OrgVolume = self.getVolume(OrgImgData)
      volumeFov = self.getVolumeFovActor(np.shape(OrgImgData))

      self.ui.ren1.AddVolume(OrgVolume)
      self.ui.ren1.AddActor(volumeFov)
      self.ui.ren1.AddActor(self.getAxes())
      self.ui.ren1.SetViewport(0, 0, 0.5, 1)
      self.ui.ren1.GetActiveCamera().Azimuth(5)
      self.ui.ren1.GetActiveCamera().Elevation(315)
      self.ui.ren1.ResetCamera()

   def ren2(self, ImgData):

      self.ui.vtkWidget.GetRenderWindow().RemoveRenderer(self.ui.ren2)
      self.ui.ren2 = vtk.vtkRenderer()
      self.ui.vtkWidget.GetRenderWindow().AddRenderer(self.ui.ren2)
      self.ui.ren2.SetViewport(0.5, 0, 1, 1)
      self.ui.ren2.SetBackground(vtk.vtkNamedColors().GetColor3d('Black'))

      volume = self.getVolume(ImgData)
      volumeFov = self.getVolumeFovActor(np.shape(ImgData))
      self.ui.ren2.AddVolume(volume)
      self.ui.ren2.AddActor(volumeFov)
      self.ui.ren2.AddActor(self.getAxes())
      self.ui.ren2.AddActor(self.getText())
      self.ui.ren2.SetViewport(0.5, 0, 1, 1)
      self.ui.ren2.GetActiveCamera().Azimuth(5)
      self.ui.ren2.GetActiveCamera().Elevation(315)
      self.ui.ren2.ResetCamera()


   def __getCurrentChart(self):
      page=self.ui.tabWidget.currentIndex() 
      if page ==0:
         chart=self.ui.chartViewBar.chart()
      elif page ==1:
         chart=self.ui.chartViewStackedBar.chart()
      elif page ==2:
         chart=self.ui.chartViewPercentBar.chart()
      else:
         chart=self.ui.chartViewPie.chart()
      return chart

   def getImgData(self):
      tt = ast.literal_eval(self.ui.ParticleTemEdit.text())
      dt = ast.literal_eval(self.ui.ParticleDiaEdit.text()) * 1e-9
      ms = ast.literal_eval(self.ui.ParticleSatEdit.text())
      cn = ast.literal_eval(self.ui.ParticleConEdit.text()) * 1e-3

      gx = ast.literal_eval(self.ui.HardSelGraXEdit.text())
      gy = ast.literal_eval(self.ui.HardSelGraYEdit.text())
      gz = ast.literal_eval(self.ui.HardSelGraZEdit.text())

      fx = ast.literal_eval(self.ui.HardDriFreXEdit.text()) * 1e3
      fy = ast.literal_eval(self.ui.HardDriFreYEdit.text()) * 1e3
      fz = ast.literal_eval(self.ui.HardDriFreZEdit.text()) * 1e3

      ax = ast.literal_eval(self.ui.HardDriAmpXEdit.text()) * 1e-3
      ay = ast.literal_eval(self.ui.HardDriAmpYEdit.text()) * 1e-3
      az = ast.literal_eval(self.ui.HardDriAmpZEdit.text()) * 1e-3

      rt = ast.literal_eval(self.ui.HardRepTimeEdit.text()) * 1e-3
      sf = ast.literal_eval(self.ui.HardSamFreEdit.text()) * 1e6

      ref = self.ui.ParticleRelchkBox.isChecked()
      re = 0
      if ref == True:
         re = ast.literal_eval(self.ui.ParticleRelEdit.text()) * 1e-6

      nsf = self.ui.NoisechkBox.isChecked()
      ns = 0
      if nsf == True:
         ns = ast.literal_eval(self.ui.NoiseEdit.text())

      bgf = self.ui.BackgroundchkBox.isChecked()
      bg = 0
      if bgf == True:
         bg = ast.literal_eval(self.ui.BackgroundEdit.text())

      ImgStru, OrgImgData, self.Message = get_ReconImg(tt, dt, ms, cn, gx, gy, gz, fx, fy, fz, ax, ay, az, rt, sf, ref, re, nsf, ns,
                                         bgf, bg, self.ui.comboTheme.currentIndex())
      OrgImgData = OrgImgData / np.max(OrgImgData)
      return  ImgStru.get_ImagSiganl()[1][0], OrgImgData

   @pyqtSlot()
   def on_toolBtn_GenData_clicked(self):

      self.ImgData=None
      self.OrgImgData=None

      self.ImgData,self.OrgImgData=self.getImgData()

      bar = myProgressBar()
      bar.setValue(0, 2, TDVISUALREND)
      self.ren1(self.OrgImgData)

      bar.setValue(1, 2, TDVISUALREND)
      self.ren2(self.ImgData)
      self.ui.iren.Initialize()
      bar.setValue(2, 2, TDVISUALREND)
      bar.close()

      self.ui.ProjectionView.figure.clear()
      ax1 = self.ui.ProjectionView.figure.add_subplot(1, 1, 1, label="points")
      l = np.shape(self.ImgData)[0]
      r = np.shape(self.ImgData)[1]
      self.ui.ImageSliceslider.setVisible(False)
      self.ui.ImageSlicelabel.setVisible(False)

      Ixy = np.zeros((l, r))
      for i in range(l):
         for j in range(r):
            Ixy[i, j] = max(self.ImgData[i, j, :])
      Ixy = Ixy / np.max(Ixy)
      ax1.imshow(Ixy, cmap=plt.get_cmap('binary'))
      ax1.set_title("X-Y")
      ax1.axis("off")
      self.ui.ProjectionView.redraw()

   @pyqtSlot()
   def on_toolBtn_Counting_clicked(self):
      dirpath, ok = QFileDialog.getSaveFileName(self, 'select save path', '', 'json(*.json)') #C:/
      if ok:
         with open(dirpath[0], 'w', encoding='utf-8') as json_file:
            json.dump(self.Message, json_file, cls=JsonDefaultEnconding)

   @pyqtSlot(int)
   def on_comboTheme_currentIndexChanged(self,index):
      self.OrgImgData = get_OriImgData(index)
      self.ren1(self.OrgImgData)

      if self.ImgData is not None:
         self.ui.vtkWidget.GetRenderWindow().RemoveRenderer(self.ui.ren2)
         self.ui.ren2 = vtk.vtkRenderer()
         self.ui.vtkWidget.GetRenderWindow().AddRenderer(self.ui.ren2)
         self.ui.ren2.SetViewport(0.5, 0, 1, 1)
         self.ui.ren2.SetBackground(vtk.vtkNamedColors().GetColor3d('Black'))

      self.ui.iren.Initialize()
      self.ui.ProjectionView.figure.clear()
      self.ui.ProjectionView.redraw()



   @pyqtSlot(int)
   def on_comboAnimation_currentIndexChanged(self,index):
      chart=self.__getCurrentChart()
      chart.setAnimationOptions(QChart.AnimationOption(index))

   @pyqtSlot()

   def getVolumeFovActor(self, N):
      xsize = N[0]
      ysize = N[1]
      zsize = N[2]

      x = [(0.0, 0.0, 0.0), (xsize, 0.0, 0.0), (xsize, ysize, 0.0), (0.0, ysize, 0.0),
           (0.0, 0.0, 0.0),
           (0.0, 0.0, zsize), (xsize, 0.0, zsize), (xsize, 0.0, 0.0), (xsize, 0.0, zsize),
           (xsize, ysize, zsize), (xsize, ysize, 0.0), (xsize, ysize, zsize),
           (0.0, ysize, zsize), (0.0, ysize, 0.0), (0.0, ysize, zsize),
           (0.0, 0.0, zsize)]

      points = vtk.vtkPoints()
      for i in x:
         points.InsertNextPoint(i)

      polyLine = vtk.vtkPolyLine()
      polyLine.GetPointIds().SetNumberOfIds(16)
      for i in range(0, 16):
         polyLine.GetPointIds().SetId(i, i)

      cells = vtk.vtkCellArray()
      cells.InsertNextCell(polyLine)

      polyData = vtk.vtkPolyData()

      polyData.SetPoints(points)

      polyData.SetLines(cells)

      mapper = vtk.vtkPolyDataMapper()
      mapper.SetInputData(polyData)
      actor = vtk.vtkActor()
      actor.SetMapper(mapper)
      actor.GetProperty().SetColor(vtk.vtkNamedColors().GetColor3d("Banana"))

      return actor

   def getVolume(self,ImgData):

      N = (np.shape(ImgData)[0], np.shape(ImgData)[1], np.shape(ImgData)[2])
      vol = vtk.vtkStructuredPoints()
      vol.SetDimensions(N[0], N[1], N[2])
      x0 = 0
      y0 = 0
      z0 = 0
      vol.SetOrigin(x0, y0, z0)
      sp = 1
      vol.SetSpacing(sp, sp, sp)
      scalars = vtk.vtkDoubleArray()
      scalars.SetNumberOfComponents(1)
      scalars.SetNumberOfTuples(N[0] * N[1] * N[2])
      for k in range(0, N[0]):
         for j in range(0, N[1]):
            for i in range(0, N[2]):
               offset = i * N[0] * N[1] + j * N[0] + k
               s=ImgData[k, j, i]*ImgData[k, j, i]*ImgData[k, j, i]
               scalars.InsertTuple1(offset, s)

      vol.GetPointData().SetScalars(scalars)

      opacityTransferFunction = vtk.vtkPiecewiseFunction()
      opacityTransferFunction.AddPoint(0, 0.0)
      opacityTransferFunction.AddPoint(5, 0.5)
      colorTransferFunction = vtk.vtkColorTransferFunction()
      colorTransferFunction.AddRGBPoint(0, 1, 1, 1)
      colorTransferFunction.AddRGBPoint(255, 0, 0, 0)

      volumeProperty = vtk.vtkVolumeProperty()
      volumeProperty.SetColor(colorTransferFunction)
      volumeProperty.SetScalarOpacity(opacityTransferFunction)
      volumeProperty.SetInterpolationTypeToLinear()

      volumeMapper = vtk.vtkFixedPointVolumeRayCastMapper()
      volumeMapper.SetInputData(vol)

      volume = vtk.vtkVolume()
      volume.SetMapper(volumeMapper)
      volume.SetProperty(volumeProperty)

      return volume

   def getAxes(self):
      axes = vtk.vtkAxesActor()
      axes.SetPosition(0, 0, 0)
      axes.SetTotalLength(10, 10, 10)
      axes.SetXAxisLabelText("x")
      axes.SetYAxisLabelText("y")
      axes.SetZAxisLabelText("z")

      xAxisCaptionActor = axes.GetXAxisCaptionActor2D()
      xAxisCaptionActor.GetTextActor().SetTextScaleModeToNone()
      xAxisCaptionActor.GetCaptionTextProperty().SetFontSize(10)
      xAxisCaptionActor.GetCaptionTextProperty().SetColor(vtk.vtkNamedColors().GetColor3d('Red'))

      yAxisCaptionActor = axes.GetYAxisCaptionActor2D()
      yAxisCaptionActor.GetTextActor().SetTextScaleModeToNone()
      yAxisCaptionActor.GetCaptionTextProperty().SetFontSize(10)
      yAxisCaptionActor.GetCaptionTextProperty().SetColor(vtk.vtkNamedColors().GetColor3d('Green'))

      zAxisCaptionActor = axes.GetZAxisCaptionActor2D()
      zAxisCaptionActor.GetTextActor().SetTextScaleModeToNone()
      zAxisCaptionActor.GetCaptionTextProperty().SetFontSize(10)
      zAxisCaptionActor.GetCaptionTextProperty().SetColor(vtk.vtkNamedColors().GetColor3d('Blue'))

      return axes

   def getText(self):
      textProperty = vtk.vtkTextProperty()
      textProperty.SetFontSize(10)
      textProperty.SetJustificationToCentered()
      textProperty.SetColor(vtk.vtkNamedColors().GetColor3d('White'))  # LightGoldenrodYellow
      textmapper = vtk.vtkTextMapper()

      xmax = str(2000 * self.Message[DRIVEFIELD][XDIRECTIOND][0] / self.Message[SELECTIONFIELD][XGRADIENT])[0:5]
      ymax = str(2000 * self.Message[DRIVEFIELD][YDIRECTIOND][0] / self.Message[SELECTIONFIELD][YGRADIENT])[0:5]
      zmax = str(2000 * self.Message[DRIVEFIELD][ZDIRECTIOND][0] / self.Message[SELECTIONFIELD][ZGRADIENT])[0:5]

      xr = str(np.shape(self.ImgData)[0])
      yr = str(np.shape(self.ImgData)[1])
      zr = str(np.shape(self.ImgData)[2])
      strvalue="FOV(cm): "+xmax+" , "+ymax+" , "+zmax+"\n\nResolution: "+xr+" X "+yr+" X "+zr+"\n\nSSIM: "+str(ssim(self.OrgImgData[1:-1,1:-1,1:-1], self.ImgData,data_range=1,channel_axis=None))[0:6]
      textmapper.SetInput(strvalue)
      textmapper.SetTextProperty(textProperty)

      textactor = vtk.vtkActor2D()
      textactor.SetMapper(textmapper)
      textactor.GetActualPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
      textactor.GetPositionCoordinate().SetValue(0.85, 0.85)

      return textactor

   @pyqtSlot()
   def on_InterferenceBtn_clicked(self):
      if self.Message is not None:
         ref = self.ui.ParticleRelchkBox.isChecked()
         re = 0
         if ref == True:
            re = ast.literal_eval(self.ui.ParticleRelEdit.text()) * 1e-6

         nsf = self.ui.NoisechkBox.isChecked()
         ns = 0
         if nsf == True:
            ns = ast.literal_eval(self.ui.NoiseEdit.text())

         bgf = self.ui.BackgroundchkBox.isChecked()
         bg = 0
         if bgf == True:
            bg = ast.literal_eval(self.ui.BackgroundEdit.text())

         bar = myProgressBar()
         ImgStru = get_ReconImgItf(ref, re, nsf, ns, bgf, bg, self.Message)
         self.ImgData = ImgStru.get_ImagSiganl()[1][0]
         bar.setValue(0, 1, TDVISUALREND)
         self.ren2(self.ImgData)
         self.ui.iren.Initialize()
         bar.setValue(1, 1, TDVISUALREND)
         bar.close()

   @pyqtSlot()
   def on_btnBuildStackedBar_clicked(self):
      self.ui.ProjectionView.figure.clear()
      ax1 = self.ui.ProjectionView.figure.add_subplot(1, 1, 1, label="points")
      l = np.shape(self.ImgData)[0]
      r = np.shape(self.ImgData)[1]
      o = np.shape(self.ImgData)[2]
      index=self.ui.comboThemePlane.currentIndex()
      if index==0:
         Ixy = np.zeros((l, r))
         for i in range(l):
            for j in range(r):
               Ixy[i, j] = max(self.ImgData[i, j, :])
         Ixy = Ixy / np.max(Ixy)
         ax1.imshow(Ixy,cmap=plt.get_cmap('binary'))
         ax1.set_title("X-Y")
         ax1.axis("off")
      elif index==1:
         Ixy = np.zeros((l, r))
         Ixy[:, :] = self.ImgData[:, :, int(self.ui.ImageSliceslider.value())]
         Ixy = Ixy / np.max(Ixy)
         ax1.imshow(Ixy, cmap=plt.get_cmap('binary'))
         ax1.set_title("X-Y")
         ax1.axis("off")
      elif index==2:
         Ixz = np.zeros((l, o))
         for i in range(l):
            for j in range(o):
               Ixz[i, j] = max(self.ImgData[i, :, j])
         Ixz = Ixz / np.max(Ixz)
         ax1.imshow(Ixz, cmap=plt.get_cmap('binary'))
         ax1.set_title("X-Z")
         ax1.axis("off")
      elif index==3:
         Ixz = np.zeros((l, o))
         Ixz[:, :] = self.ImgData[:, int(self.ui.ImageSliceslider.value()), :]
         Ixz = Ixz / np.max(Ixz)
         ax1.imshow(Ixz, cmap=plt.get_cmap('binary'))
         ax1.set_title("X-Z")
         ax1.axis("off")
      elif index==4:
         Iyz = np.zeros((r, o))
         for i in range(r):
            for j in range(o):
               Iyz[i, j] = max(self.ImgData[:, i, j])
         Iyz = Iyz / np.max(Iyz)
         ax1.imshow(Iyz, cmap=plt.get_cmap('binary'))
         ax1.set_title("Y-Z")
         ax1.axis("off")
      elif index==5:
         Iyz = np.zeros((r, o))
         Iyz[:, :] = self.ImgData[int(self.ui.ImageSliceslider.value()), :, :]
         Iyz = Iyz / np.max(Iyz)
         ax1.imshow(Iyz, cmap=plt.get_cmap('binary'))
         ax1.set_title("Y-Z")
         ax1.axis("off")

      self.ui.ProjectionView.redraw()

   @pyqtSlot(int)
   def on_comboThemeProc_currentIndexChanged(self, index):
      if index==1:
         self.ui.comboThemeThds.setVisible(True)
         self.ui.ComThdsLabel.setVisible(True)
         SET_VALUE('PROFLAG', 1)
         SET_VALUE('PROTHDS', 4)
      else:
         self.ui.comboThemeThds.setVisible(False)
         self.ui.ComThdsLabel.setVisible(False)
         SET_VALUE('PROFLAG', 4)

      self.ui.ProjectionView.redraw()

   @pyqtSlot(int)
   def on_comboThemeThds_currentIndexChanged(self, index):
      if index==0:
         SET_VALUE('PROTHDS', 4)
      if index==1:
         SET_VALUE('PROTHDS', 8)
      if index==2:
         SET_VALUE('PROTHDS', 16)

   @pyqtSlot(int)
   def on_comboThemePlane_currentIndexChanged(self, index):
      if self.ImgData is None:
         return
      self.ui.ProjectionView.figure.clear()
      ax1 = self.ui.ProjectionView.figure.add_subplot(1, 1, 1, label="points")
      l = np.shape(self.ImgData)[0]
      r = np.shape(self.ImgData)[1]
      o = np.shape(self.ImgData)[2]
      if index == 0:
         self.ui.ImageSliceslider.setVisible(False)
         self.ui.ImageSlicelabel.setVisible(False)
         Ixy = np.zeros((l, r))
         for i in range(l):
            for j in range(r):
               Ixy[i, j] = max(self.ImgData[i, j, :])
         Ixy = Ixy / np.max(Ixy)
         ax1.imshow(Ixy, cmap=plt.get_cmap('binary'))
         ax1.set_title("X-Y")
         ax1.axis("off")
         self.ui.ProjectionView.redraw()

      if index == 2:
         self.ui.ImageSliceslider.setVisible(False)
         self.ui.ImageSlicelabel.setVisible(False)
         Ixz = np.zeros((l, o))
         for i in range(l):
            for j in range(o):
               Ixz[i, j] = max(self.ImgData[i, :, j])
         Ixz = Ixz / np.max(Ixz)
         ax1.imshow(Ixz, cmap=plt.get_cmap('binary'))
         ax1.set_title("X-Z")
         ax1.axis("off")
         self.ui.ProjectionView.redraw()

      if index == 4:
         self.ui.ImageSliceslider.setVisible(False)
         self.ui.ImageSlicelabel.setVisible(False)
         Iyz = np.zeros((r, o))
         for i in range(r):
            for j in range(o):
               Iyz[i, j] = max(self.ImgData[:, i, j])
         Iyz = Iyz / np.max(Iyz)
         ax1.imshow(Iyz, cmap=plt.get_cmap('binary'))
         ax1.set_title("Y-Z")
         ax1.axis("off")
         self.ui.ProjectionView.redraw()


      if index == 1:
         self.ui.ImageSliceslider.setVisible(True)
         self.ui.ImageSlicelabel.setVisible(True)
         self.ui.ImageSliceslider.setMaximum(np.shape(self.ImgData)[2]-1)
         self.ui.ImageSlicelabel.setText("第1层")
         Ixy = np.zeros((l, r))
         Ixy[:, :] = self.ImgData[:, :, 0]
         Ixy = Ixy / np.max(Ixy)
         ax1.imshow(Ixy, cmap=plt.get_cmap('binary'))
         ax1.set_title("X-Y")
         ax1.axis("off")
         self.ui.ProjectionView.redraw()

      if index == 3:
         self.ui.ImageSliceslider.setVisible(True)
         self.ui.ImageSlicelabel.setVisible(True)
         self.ui.ImageSliceslider.setMaximum(np.shape(self.ImgData)[1]-1)
         self.ui.ImageSlicelabel.setText("第1层")
         Ixz = np.zeros((l, o))
         Ixz[:, :] = self.ImgData[:, 0, :]
         Ixz = Ixz / np.max(Ixz)
         ax1.imshow(Ixz, cmap=plt.get_cmap('binary'))
         ax1.set_title("X-Z")
         ax1.axis("off")
         self.ui.ProjectionView.redraw()
      if index == 5:
         self.ui.ImageSliceslider.setVisible(True)
         self.ui.ImageSlicelabel.setVisible(True)
         self.ui.ImageSliceslider.setMaximum(np.shape(self.ImgData)[0]-1)
         self.ui.ImageSlicelabel.setText("第1层")
         Iyz = np.zeros((r, o))
         Iyz[:, :] = self.ImgData[0, :, :]
         Iyz = Iyz / np.max(Iyz)
         ax1.imshow(Iyz, cmap=plt.get_cmap('binary'))
         ax1.set_title("Y-Z")
         ax1.axis("off")
         self.ui.ProjectionView.redraw()