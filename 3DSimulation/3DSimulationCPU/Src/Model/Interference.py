import numpy as np
from MPIRF.Config.ConstantList import *

import math

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
