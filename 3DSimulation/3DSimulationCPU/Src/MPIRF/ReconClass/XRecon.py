# coding=UTF-8
import numpy as np

from MPIRF.ReconClass.BaseClass.ReconBase import *
from MPIRF.Config.ConstantList import *
from scipy.interpolate import griddata
from Model.Interference import *

import csv

'''
XRecon.py: The XRecon Class reconstruct the MPI image signal based on the x-space method.
'''

class XReconClass(ReconBaseClass):

    def __init__(self,Message):
        super().__init__()
        self.Rel=Message[EXTENDED]["Relaxation"]
        self.RelT = Message[EXTENDED]["RelaxationTime"]
        self.Noise=Message[EXTENDED]["NoiseFlag"]
        self.Background=Message[EXTENDED]["BackgroundFlag"]
        if self.Noise:
            self.NoiseValue=Message[EXTENDED]["NoiseValue"]
        if self.Background:
            self.BackgroundValue=Message[EXTENDED]["BackgroundValue"]

        self.Fn = Message[SAMPLE][SAMNUMBER]
        self.Fs = Message[SAMPLE][FREQUENCY]
        self.Rt = Message[DRIVEFIELD][REPEATTIME]
        # self._ffv2 = Message[EXTENDED]["FFV2"]

        self._ImageRecon(Message)

    # Call image reconstruction algorithm and Resize image.
    def _ImageRecon(self, Message):
        self._ImagSignal.append(self.__XSpace(Message[MEASUREMENT][MEASIGNAL], Message[MEASUREMENT][AUXSIGNAL]))
        self._ImagSignal.append(self._ImageReshape(Message[EXTENDED][RFFP],Message[EXTENDED][STEP]))

        return True

    # Resize image.
    def _ImageReshape(self, Rffp, Step):
        pointx = np.arange(min(Rffp[0][:]), max(Rffp[0][:]) + Step, Step)
        pointy = np.arange(min(Rffp[1][:]), max(Rffp[1][:]) + Step, Step)
        pointz = np.arange(min(Rffp[2][:]), max(Rffp[2][:]) + Step, Step)
        xpos, ypos , zpos= np.meshgrid(pointy, pointx, pointz)
        ImgTan = griddata((Rffp[1], Rffp[0], Rffp[2]), self._ImagSignal[0], (xpos, ypos, zpos), method='linear')

        temp=np.isnan(ImgTan[1:-1, 1:-1, 1:-1])
        if True in temp:
            ImgTan = griddata((Rffp[1], Rffp[0], Rffp[2]), self._ImagSignal[0], (xpos, ypos, zpos), method='nearest')

        ImgTan = ImgTan[1:-1, 1:-1, 1:-1]
        ImgTan = ImgTan / np.max(ImgTan)
        return [ImgTan]

    #x-space algorithm
    def __XSpace(self,U, Vffp):

        temp = Vffp ** 2
        VffpLen = np.sqrt(temp[0] + temp[1] + temp[2])
        VffpDir = np.divide(Vffp, np.tile(VffpLen, (3, 1)))

        temp = np.transpose(U) * VffpDir
        SigTan = temp[0] + temp[1] + temp[2]

        if self.Rel:
            SigTan=InterferenceClass().RelaxationCPU(SigTan,self.RelT, self.Fn, self.Fs, self.Rt)

        if self.Noise:
            tempn=np.transpose(self.NoiseValue) * VffpDir
            noise=tempn[0] + tempn[1] + tempn[2]
            SigTan=SigTan+noise
        if self.Background:
            tempb = np.transpose(self.BackgroundValue) * VffpDir
            background = tempb[0] + tempb[1] + tempb[2]
            SigTan = SigTan + background

        #########################################################################################
        # temp = self._ffv2 ** 2
        # VffpLen = np.sqrt(temp[0] + temp[1] + temp[2])
        #########################################################################################

        ImgTan = SigTan / VffpLen
        ImgTan = ImgTan / np.max(ImgTan)
        return ImgTan