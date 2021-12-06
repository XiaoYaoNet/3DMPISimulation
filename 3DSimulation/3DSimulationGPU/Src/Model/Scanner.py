from pylab import *

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from concurrent.futures import ThreadPoolExecutor, wait,ALL_COMPLETED,as_completed,FIRST_COMPLETED

from UI.myProgressBar import *
from Model.Phantom import *

from MPIRF.Config.UIConstantListEn import SIMUVOL

from MPIRF.Config.ConstantList import GET_VALUE
from MPIRF.DataClass.BassClass.DataBase import *

mod = SourceModule(r"""
        #include <cmath>
        __global__ void doublify(double* Phantom, double* H, double* Dhx, double Bx, double Cx, double* Dhy, double By, double Cy, double* Dhz, double Bz, double Cz, double B, int length, int width, int height,int step, int start)
        {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            int z = blockIdx.z * blockDim.z + threadIdx.z;
            double Ax=0;
            double Ay=0;
            double Az=0;
            int index2=0;
            int index3=0;
            double temp=0;
            if (x < length && y < width && z < height ) {
                int index1 = z + y*height + x*width*height;
                for(int i=0;i<step;i++)
                {
                    index2=i*length*width*height+x+y*length+z*width*length;
                    index3=i+start;
                    Ax = Dhx[index3] - Bx*((x)*(1e-4) - Cx);
                    Ay = Dhy[index3] - By*((y)*(1e-4) - Cy);
                    Az = Dhz[index3] - Bz*((z)*(1e-4) - Cz);

                    temp=sqrt(Ax*Ax + Ay*Ay + Az*Az);

                    temp=temp*B;
                    if(temp!=0)
                    {
                        temp=(1/(temp*temp))-(1/(sinh(temp)*sinh(temp)));
                    }else{
                        temp=1/3;
                    }
                    H[index2]=temp*Phantom[index1];
                }
            }
        }
""")

doublify = mod.get_function("doublify")


def Work( H, i, hStep, U, Coeff):
    for j in range(hStep):
        result = np.sum(H[j, :, :, :])
        U[i + j, 0] = Coeff[0, i + j] * result
        U[i + j, 1] = Coeff[1, i + j] * result
        U[i + j, 2] = Coeff[2, i + j] * result

    return True

def ThdFunc(DH,l,r,o,GSc,B,C):
    DHt=np.tile(DH, (l, r, o, 1))
    test=np.subtract(DHt,GSc)
    H=np.sqrt(test[:,:,:,2]**2+test[:,:,:,1]**2+test[:,:,:,0]**2)
    DLF = (1 / ((B * H) ** 2)) - (1 / ((np.sinh(B * H)) ** 2))

    s = C * DLF
    signal=np.sum(s[:,:,:])

    return signal

def CudaKernal(C,
               DHx, DHy, DHz,
               Steph,
               Nx, Ny, Nz,
               Fn, Bb,
               Gx, Gy, Gz,
               Maxx, Maxy, Maxz,
               U,Coeff):

    bar=myProgressBar()
    tn=Fn+1

    C_gpu = cuda.mem_alloc(C.astype(np.float64).nbytes)
    cuda.memcpy_htod(C_gpu, C.astype(np.float64))

    DHtx_gpu = cuda.mem_alloc(DHx.astype(np.float64).nbytes)
    cuda.memcpy_htod(DHtx_gpu, DHx.astype(np.float64))

    DHty_gpu = cuda.mem_alloc(DHy.astype(np.float64).nbytes)
    cuda.memcpy_htod(DHty_gpu, DHy.astype(np.float64))

    DHtz_gpu = cuda.mem_alloc(DHz.astype(np.float64).nbytes)
    cuda.memcpy_htod(DHtz_gpu, DHz.astype(np.float64))

    Htemp = np.zeros((Steph, Nx, Ny, Nz))
    Htemp = Htemp.astype(np.float64)
    H_gpu = cuda.mem_alloc(Htemp.nbytes)
    cuda.memcpy_htod(H_gpu, Htemp)

    executor = ThreadPoolExecutor(max_workers=10)
    all_task = []

    nTheads = 8
    nBlockx = math.ceil(Nx / nTheads)
    nBlocky = math.ceil(Ny / nTheads)
    nBlockz = math.ceil(Nz / nTheads)

    i = 0
    j = 0
    while i < Fn:

        if (i + Steph) >= Fn:
            Steph = Fn - i
            Htemp = np.zeros((Steph, Nx, Ny, Nz))
            Htemp = Htemp.astype(np.float64)
            H_gpu = cuda.mem_alloc(Htemp.nbytes)
            cuda.memcpy_htod(H_gpu, Htemp)

        doublify(
            C_gpu, H_gpu,
            DHtx_gpu, np.float64(Gx), np.float64(Maxx),
            DHty_gpu, np.float64(Gy), np.float64(Maxy),
            DHtz_gpu, np.float64(Gz), np.float64(Maxz),
            np.float64(Bb),
            np.int32(Nx), np.int32(Ny), np.int32(Nz), np.int32(Steph), np.int32(i),
            block=(nTheads, nTheads, nTheads), grid=(nBlockx, nBlocky, nBlockz))

        cuda.memcpy_dtoh(Htemp, H_gpu)

        temp=sum(Htemp)

        args = [Htemp, i, Steph, U, Coeff]
        all_task.append(executor.submit(lambda p: Work(*p), args))
        j = j + 1
        i = i + Steph
        bar.setValue(i+1, tn, SIMUVOL)

    wait(all_task, return_when=ALL_COMPLETED)
    executor.shutdown()
    bar.setValue(tn, tn, SIMUVOL)
    bar.close()

    return True

class ScannerClass(DataBaseClass):

    def __init__(self,
                 VirtualPhantom,
                 SelectGradietX = 2.4,
                 SelectGradietY = 4.8,
                 SelectGradietZ = 2.4,
                 DriveFrequencyX = 26.04e3,
                 DriveFrequencyY = 24.51e3,
                 DriveFrequencyZ = 21.54e3,
                 DriveAmplitudeX = 12e-3,
                 DriveAmplitudeY = 12e-3,
                 DriveAmplitudeZ = 12e-3,
                 RepetitionTime = 21.54e-3,
                 SampleFrequency = 2.5e5):

        super().__init__()

        self._VirtualPhantom = VirtualPhantom

        self._CoilSensitivity = 1.0

        self._Gx = SelectGradietX/U0
        self._Gy = SelectGradietY/U0
        self._Gz = SelectGradietZ/U0
        self._Gg = np.array([[self._Gx], [self._Gy], [self._Gz]])

        self._Ax = DriveAmplitudeX/U0
        self._Ay = DriveAmplitudeY/U0
        self._Az = DriveAmplitudeZ/U0

        self._Fx = DriveFrequencyX
        self._Fy = DriveFrequencyY
        self._Fz = DriveFrequencyZ

        self._Rt = RepetitionTime
        self._Sf = SampleFrequency

        self._Maxx = self._Ax / self._Gx
        self._Maxy = self._Ay / self._Gy
        self._Maxz = self._Az / self._Gz

        self._T = np.arange((1 / self._Sf), self._Rt, (1 / self._Sf))
        self._Fn = np.shape(self._T)[0]

        self._DHx, self._DeriDHx = self.__DriveStrength(self._Ax, self._Fx, self._T)
        self._DHy, self._DeriDHy = self.__DriveStrength(self._Ay, self._Fy, self._T)
        self._DHz, self._DeriDHz = self.__DriveStrength(self._Az, self._Fz, self._T)

        self._DH = np.array([self._DHx, self._DHy, self._DHz])
        self._DeriDH = np.array([self._DeriDHx, self._DeriDHy, self._DeriDHz])

        self._Ffpr = np.divide(self._DH, np.tile(self._Gg, (1, np.shape(self._DH)[1])))


        self._Ffpx = self._Ffpr[0]
        self._Ffpy = self._Ffpr[1]
        self._Ffpz = self._Ffpr[2]

        ##################################################################################################
        # self._DHx2, self._DeriDHx2 = self.__DriveStrength(self._Ax, self._Fx, self._T-0.75e-6)
        # self._DHy2, self._DeriDHy2 = self.__DriveStrength(self._Ay, self._Fy, self._T-0.75e-6)
        # self._DHz2, self._DeriDHz2 = self.__DriveStrength(self._Az, self._Fz, self._T-0.75e-6)
        #
        # self._DH2 = np.array([self._DHx, self._DHy, self._DHz])
        # self._DeriDH2 = np.array([self._DeriDHx2, self._DeriDHy, self._DeriDHz2])
        #
        # self._Ffpv2 = np.divide(self._DeriDH2, np.tile(self._Gg, (1, np.shape(self._DeriDH2)[1])))
        ##################################################################################################

        self._Step = 1e-4
        self._Nx = len( np.arange(min(self._Ffpx[:]), max(self._Ffpx[:]) + self._Step, self._Step) )
        self._Ny = len( np.arange(min(self._Ffpy[:]), max(self._Ffpy[:]) + self._Step, self._Step) )
        self._Nz = len( np.arange(min(self._Ffpz[:]), max(self._Ffpz[:]) + self._Step, self._Step) )

        self._PhantomMatrix = self._VirtualPhantom.getShape(self._Nx, self._Ny, self._Nz)

        self._Ffpv = np.divide(self._DeriDH, np.tile(self._Gg, (1, np.shape(self._DeriDH)[1])))

        if GET_VALUE('PROFLAG')==1:
            self._get_Voltage_CPU()
        else:
            self._get_Voltage_GPU()

        self._init_Message()

    def __DriveStrength(self, DriveAmplitude, DriveFrequency, TSquence):
        DHx = DriveAmplitude * np.cos(2.0 * PI * DriveFrequency * TSquence + PI / 2.0) * (-1.0)
        DeriDHx = DriveAmplitude * np.sin(2.0 * PI * DriveFrequency * TSquence + PI / 2.0) * 2.0 * PI * DriveFrequency
        return DHx, DeriDHx

    def _get_Voltage_GPU(self):
        self._Voltage = np.zeros((self._Fn, 3))
        self._Coeff = self._CoilSensitivity * self._VirtualPhantom._Mm * self._VirtualPhantom._Bb * self._DeriDH
        self._Stemph=1

        CudaKernal(self._PhantomMatrix,
                   self._DHx,self._DHy,self._DHz,
                   self._Stemph,
                   self._Nx,self._Ny,self._Nz,
                   self._Fn,self._VirtualPhantom._Bb,
                   self._Gx,self._Gy,self._Gz,
                   self._Maxx,self._Maxy,self._Maxz,
                   self._Voltage, self._Coeff)

        return True

    def _get_Voltage_CPU(self):
        bar = myProgressBar()
        tn = self._Fn + self._Nx*self._Ny*self._Nz
        num=0
        self._Voltage = np.zeros((self._Fn, 3))
        GSc = np.zeros((self._Nx,self._Ny,self._Nz, 3))
        for i in range(self._Nx):
            for j in range(self._Ny):
                for k in range(self._Nz):
                    x=(i)*(1e-4) - self._Maxx
                    y=(j)*(1e-4) - self._Maxy
                    z=(k)*(1e-4) - self._Maxz
                    temp=np.multiply(self._Gg,[[x],[y],[z]])
                    GSc[i, j, k, 0]=temp[0]
                    GSc[i, j, k, 1]=temp[1]
                    GSc[i, j, k, 2]=temp[2]
                    num = num + 1
                    bar.setValue(num, tn, SIMUVOL)


        executor = ThreadPoolExecutor(max_workers=GET_VALUE('PROTHDS'))
        print(GET_VALUE('PROTHDS'))
        all_task = [executor.submit(lambda p: ThdFunc(*p), [self._DH[:,i],self._Nx,self._Ny,self._Nz,GSc,self._VirtualPhantom._Bb,self._PhantomMatrix]) for i in range(int(self._Fn))]
        wait(all_task, return_when=ALL_COMPLETED)
        Coeff=self._CoilSensitivity*self._VirtualPhantom._Bb* self._VirtualPhantom._Mm
        i=0
        for value in all_task:
            temp=value.result()
            self._Voltage[i, 0] = Coeff * self._DeriDH[0,i] * temp
            self._Voltage[i, 1] = Coeff * self._DeriDH[1,i] * temp
            self._Voltage[i, 2] = Coeff * self._DeriDH[2,i] * temp
            i=i+1
            num = num + 1
            bar.setValue(num, tn, SIMUVOL)
        executor.shutdown()
        bar.close()

    def _init_Message(self):

        self._set_MessageValue(MAGNETICPARTICL, TEMPERATURE, self._VirtualPhantom._Tt)
        self._set_MessageValue(MAGNETICPARTICL, DIAMETER, self._VirtualPhantom._Diameter)
        self._set_MessageValue(MAGNETICPARTICL, SATURATIONMAG, self._VirtualPhantom._Mm)

        self._set_MessageValue(SELECTIONFIELD, XGRADIENT, self._Gx)
        self._set_MessageValue(SELECTIONFIELD, YGRADIENT, self._Gy)
        self._set_MessageValue(SELECTIONFIELD, ZGRADIENT, self._Gz)

        self._set_MessageValue(DRIVEFIELD, XDIRECTIOND, np.array([self._Ax, self._Fx, 0]))
        self._set_MessageValue(DRIVEFIELD, YDIRECTIOND, np.array([self._Ay, self._Fy, 0]))
        self._set_MessageValue(DRIVEFIELD, ZDIRECTIOND, np.array([self._Az, self._Fz, 0]))
        self._set_MessageValue(DRIVEFIELD, REPEATTIME, self._Rt)
        self._set_MessageValue(DRIVEFIELD, WAVEFORMD, SINE)

        self._set_MessageValue(SAMPLE, TOPOLOGY, FFP)
        self._set_MessageValue(SAMPLE, FREQUENCY, self._Sf)
        self._set_MessageValue(SAMPLE, SAMNUMBER, self._Fn)
        self._set_MessageValue(SAMPLE, BEGINTIME, None)
        self._set_MessageValue(SAMPLE, SENSITIVITY, self._CoilSensitivity)

        self._set_MessageValue(MEASUREMENT, TYPE, 2)
        self._set_MessageValue(MEASUREMENT, BGFLAG, np.ones(np.shape(self._Voltage), dtype='bool'))
        self._set_MessageValue(MEASUREMENT, MEASIGNAL, self._Voltage)
        self._set_MessageValue(MEASUREMENT, AUXSIGNAL, self._Ffpv)
        self._set_MessageValue(MEASUREMENT, MEANUMBER, np.array([self._Nx, self._Ny, self._Nz], dtype='int64'))

        self.Message[EXTENDED] = {STEP: self._Step, RFFP: self._Ffpr}
        # self.Message[EXTENDED] = {STEP: self._Step, RFFP: self._Ffpr, "FFV2": self._Ffpv2}

        return True