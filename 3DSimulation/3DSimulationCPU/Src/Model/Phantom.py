import numpy as np
from MPIRF.Config.ConstantList import *

class PhantomClass(object):
    def __init__(self, Temperature=25, Diameter=50e-9, MagSaturation= 0.6, Concentration=5e7, index=0):
        self._Tt = Temperature + TDT
        self._Diameter = Diameter
        self._Volume = self.__get_ParticleVolume()
        self._MCore = MagSaturation/U0
        self._Mm = self.__get_MagMomentSaturation()
        self._Bb = self.__get_ParticleProperty()
        self._Concentration = Concentration
        self._index=index

    def __get_ParticleVolume(self):
        return (self._Diameter ** 3) * PI / 6.0

    def __get_MagMomentSaturation(self):
        return self._MCore * self._Volume

    def __get_ParticleProperty(self):
        return (U0 * self._Mm) / (KB * self._Tt)

    def get_Bcoeff(self):
        return self._Bcoeff

    def getShape(self, Xn, Yn, Zn):
        if self._index==0:
            return self.getEShape(Xn, Yn, Zn)
        elif self._index==1:
            return self.getPShape(Xn, Yn, Zn)
        else:
            return self.getEShape(Xn, Yn, Zn)

    def getEShape(self, Xn, Yn, Zn):
        C = np.zeros((Xn, Yn, Zn))
        C[int(Xn * (30 / 201)):int(Xn * (170 / 201)), 10:Yn - 10, int(Zn * (50 / 201)):int(Zn * (150 / 201))] = np.ones(
            (int(Xn * (170 / 201)) - int(Xn * (30 / 201)), Yn - 20, int(Zn * (150 / 201)) - int(Zn * (50 / 201))))
        C[int(Xn * (53 / 201)):int(Xn * (88 / 201)), 10:Yn - 10, int(Zn * (76 / 201)):int(Zn * (150 / 201))] = np.zeros(
            (int(Xn * (88 / 201)) - int(Xn * (53 / 201)), Yn - 20, int(Zn * (150 / 201)) - int(Zn * (76 / 201))))
        C[int(Xn * (112 / 201)):int(Xn * (147 / 201)), 10:Yn - 10, int(Zn * (76 / 201)):int(Zn * (150 / 201))] = np.zeros(
            (int(Xn * (147 / 201)) - int(Xn * (112 / 201)), Yn - 20, int(Zn * (150 / 201)) - int(Zn * (76 / 201))))
        return C*self._Concentration*1e-4*1e-4*1e-4

    def getPShape(self, Xn, Yn, Zn):
        C = np.zeros((Xn, Yn, Zn))
        C[int(Xn * (50 / 201)):int(Xn * (150 / 201)), int(Yn * (25 / 201)):int(Yn * (175 / 201)), 10:Zn - 10] = np.ones(
            (int(Xn * (150 / 201)) - int(Xn * (50 / 201)), int(Yn * (175 / 201)) - int(Yn * (25 / 201)), Zn - 20))
        C[int(Xn * (75 / 201)):int(Xn * (125 / 201)), int(Yn * (100 / 201)):int(Yn * (150 / 201)),
        10:Zn - 10] = np.zeros(
            (int(Xn * (125 / 201)) - int(Xn * (75 / 201)), int(Yn * (150 / 201)) - int(Yn * (100 / 201)), Zn - 20))
        C[int(Xn * (75 / 201)):int(Xn * (150 / 201)), int(Yn * (25 / 201)):int(Yn * (75 / 201)), 10:Zn - 10] = np.zeros(
            (int(Xn * (150 / 201)) - int(Xn * (75 / 201)), int(Yn * (75 / 201)) - int(Yn * (25 / 201)), Zn - 20))
        return C * self._Concentration*1e-4*1e-4*1e-4