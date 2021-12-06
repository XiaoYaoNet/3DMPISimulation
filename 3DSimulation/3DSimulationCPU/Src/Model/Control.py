from Model.Scanner import *
from Model.Phantom import *
import json

from MPIRF.ReconClass.XRecon import *
from MPIRF.Config.UIConstantListEn import IMGRECONSTR

def get_ReconImg(tt,dt,ms,cn,gx,gy,gz,fx,fy,fz,ax,ay,az,rt,sf, ref, re, nsf, ns, bgf, bg,index=0):
    P1 = PhantomClass(tt,dt,ms,cn,index)
    S1 = ScannerClass(P1,gx,gy,gz,fx,fy,fz,ax,ay,az,rt,sf)

    bar = myProgressBar()
    bar.setValue(0, 0, IMGRECONSTR)

    S1.Message[MEASUREMENT]['OriMeaSignal']=S1.Message[MEASUREMENT][MEASIGNAL]

    noise=None
    bgsignal=None
    if nsf:
        noise=InterferenceClass().GaussianNoise(S1.Message,ns)
    if bgf:
        bgsignal=InterferenceClass().BackgroundHarmonic(S1.Message,bg)

    S1.Message[EXTENDED]["NoiseFlag"]=nsf
    S1.Message[EXTENDED]["BackgroundFlag"] = bgf

    if noise is not None:
        # S1.Message[MEASUREMENT][MEASIGNAL]=S1.Message[MEASUREMENT][MEASIGNAL]+noise
        S1.Message[EXTENDED]["NoiseValue"] = noise

    if bgsignal is not None:
        # S1.Message[MEASUREMENT][MEASIGNAL]=S1.Message[MEASUREMENT][MEASIGNAL]+bgsignal
        S1.Message[EXTENDED]["BackgroundValue"]=bgsignal

    bar = myProgressBar()
    bar.setValue(0, 0, IMGRECONSTR)

    S1.Message[EXTENDED]["Relaxation"] = ref
    S1.Message[EXTENDED]["RelaxationTime"] = re

    ImgStru = XReconClass(S1.Message)
    bar.setValue(100, 0, IMGRECONSTR)
    bar.close()

    return ImgStru, S1._PhantomMatrix, S1.Message

def get_ReconImgItf(ref, re, nsf, ns, bgf, bg, Message):

    Message[MEASUREMENT][MEASIGNAL] = Message[MEASUREMENT]['OriMeaSignal']

    noise = None
    bgsignal = None
    if nsf:
        noise = InterferenceClass().GaussianNoise(Message, ns)
    if bgf:
        bgsignal = InterferenceClass().BackgroundHarmonic(Message, bg)

    Message[EXTENDED]["NoiseFlag"] = nsf
    Message[EXTENDED]["BackgroundFlag"] = bgf

    if noise is not None:
        # S1.Message[MEASUREMENT][MEASIGNAL]=S1.Message[MEASUREMENT][MEASIGNAL]+noise
        Message[EXTENDED]["NoiseValue"] = noise

    if bgsignal is not None:
        # S1.Message[MEASUREMENT][MEASIGNAL]=S1.Message[MEASUREMENT][MEASIGNAL]+bgsignal
        Message[EXTENDED]["BackgroundValue"] = bgsignal

    bar = myProgressBar()
    bar.setValue(0, 0, IMGRECONSTR)

    Message[EXTENDED]["Relaxation"] = ref
    Message[EXTENDED]["RelaxationTime"] = re

    ImgStru = XReconClass(Message)
    bar.setValue(100, 0, IMGRECONSTR)
    bar.close()

    return ImgStru

def get_OriImgData(index=0):

    P1 = PhantomClass(index=index)
    OriImgData = P1.getShape(100, 100, 50)
    return OriImgData/np.max(OriImgData)

class JsonDefaultEnconding(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, complex):
            return "{real}+{image}i".format(real=o.real, image=o.real)
        if isinstance(o, np.ndarray):
            return o.tolist()