import numpy as np
from MPIRF.Config.ConstantList import *

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import math


mod = SourceModule(r"""
        #include <cmath>
        __global__ void relaxtionfun(double *signal, double *rela, double *sr, int length)
        {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            if (x<length)
            {
                int index=length-x-1;
                for (int i=0;i<(x+1);i++)
                {
                    sr[x]=sr[x]+rela[i]*signal[index+i];
                }
            }
        }
""")

relaxtionfun = mod.get_function("relaxtionfun")

def RelaxationKernal(signal,rela,length,Fn):

    nTheads = 64
    nBlockx = math.ceil(Fn / nTheads)

    signal_gpu = cuda.mem_alloc(signal.astype(np.float64).nbytes)
    cuda.memcpy_htod(signal_gpu, signal.astype(np.float64))

    rela_gpu = cuda.mem_alloc(rela.astype(np.float64).nbytes)
    cuda.memcpy_htod(rela_gpu, rela.astype(np.float64))

    sr = np.zeros((int(Fn)))
    sr_gpu = cuda.mem_alloc(sr.astype(np.float64).nbytes)
    cuda.memcpy_htod(sr_gpu, sr.astype(np.float64))

    relaxtionfun(signal_gpu,rela_gpu,sr_gpu,np.int32(length),block=(nTheads,1,1), grid=(nBlockx,1))

    cuda.memcpy_dtoh(sr, sr_gpu)
    return sr


class InterferenceClass(object):

    def GaussianNoise(self, Message, SNR):
        signal=Message[MEASUREMENT][MEASIGNAL]
        noise = np.random.randn(signal.shape[0], signal.shape[1])
        noise = noise - np.mean(noise)  # 均值为0
        signal_power = np.linalg.norm(signal - signal.mean()) ** 2 / signal.size
        noise_variance = signal_power / np.power(10, (SNR / 20))
        noise = (np.sqrt(noise_variance) / np.std(noise)) * noise

        return noise

    def BackgroundHarmonic(self, Message, SIR):
        n=3
        signal = Message[MEASUREMENT][MEASIGNAL]
        Fn = Message[SAMPLE][SAMNUMBER]
        Fs = Message[SAMPLE][FREQUENCY]
        Rt = Message[DRIVEFIELD][REPEATTIME]
        fx = Message[DRIVEFIELD][XDIRECTIOND][0]
        fy = Message[DRIVEFIELD][YDIRECTIOND][0]
        fz = Message[DRIVEFIELD][ZDIRECTIOND][0]
        t = np.arange((1 / Fs), Rt, (1 / Fs))

        Xhar = np.fft.rfft(np.transpose(signal[:, 0])) / Fn
        Yhar = np.fft.rfft(np.transpose(signal[:, 1])) / Fn
        Zhar = np.fft.rfft(np.transpose(signal[:, 2])) / Fn

        Yx = Xhar.max() / np.power(10, (SIR / 20))
        Yy = Yhar.max() / np.power(10, (SIR / 20))
        Yz = Zhar.max() / np.power(10, (SIR / 20))
        sb = np.zeros((int(Fn), 3))
        sb[:, 0] = np.cos(2.0 * PI * n * fx * t + PI / 2.0) * (-1.0) * Yx
        sb[:, 1] = np.cos(2.0 * PI * n * fy * t + PI / 2.0) * (-1.0) * Yy
        sb[:, 2] = np.cos(2.0 * PI * n * fz * t + PI / 2.0) * (-1.0) * Yz

        return sb

    def RelaxationCPU(self, Message, T, Fn, Fs, Rt):
        signal= Message
        t = np.arange((1 / Fs), Rt, (1 / Fs))
        Rela = (1 / T) * np.exp((-1) * (t / T)) / Fs
        sr = np.zeros((int(Fn)))
        signal = np.flipud(signal)
        length = np.shape(signal)[0]
        for i in range(length):
            index = length - i - 1
            for j in range(i + 1):
                sr[i] = sr[i] + Rela[j] * signal[index + j]

        return sr

    def RelaxationGPU(self, Message, T, Fn, Fs, Rt):
        signal= Message
        t = np.arange((1 / Fs), Rt, (1 / Fs))
        Rela = (1 / T) * np.exp((-1) * (t / T)) / Fs
        signal = np.flipud(signal)
        length = np.shape(signal)[0]
        sr=RelaxationKernal(signal,Rela,length,Fn)

        return sr
