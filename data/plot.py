import wave
obj=wave.open("temp.wav",'rb')
print("通道数：",obj.getnchannels())
print("采样字节宽度：",obj.getsampwidth())
print("采样频率：",obj.getframerate())
print("总帧数：",obj.getnframes())
import scipy.io.wavfile as wav
import numpy as np
import pylab as pl
'''
fs,x=wav.read("temp.wav")
x=x/(256*256)
time=np.arange(0,obj.getnframes())*(1.0/fs)
pl.subplot(211)
pl.title("Before VAD")
pl.plot(time,x)
'''
#pl.xlabel("time(seconds)")
#pl.show()
'''
from data.example import myvad
paths=myvad("temp.wav",3)
print(paths)
'''
'''
obj=wave.open("vad.wav",'rb')
fs,x=wav.read("vad.wav")
x=x/(256*256)
time=np.arange(0,obj.getnframes())*(1.0/fs)
pl.subplot(212)
pl.title("After VAD")
pl.plot(time,x)
pl.xlabel("time(seconds)")
pl.subplots_adjust(wspace =0, hspace =0.4)
pl.show()

'''
'''
import matplotlib.pyplot as plt
print("plotting spectrogram...")
fs,x=wav.read("vad.wav")
framelength = 0.040 #帧长20~30ms
framesize = framelength*16000 #每帧点数 N = t*fs,通常情况下值为256或512,要与NFFT相等\
                                    #而NFFT最好取2的整数次方,即framesize最好取的整数次方
#找到与当前framesize最接近的2的正整数次方
nfftdict = {}
lists = [32,64,128,256,512,1024]
for i in lists:
    nfftdict[i] = abs(framesize - i)
sortlist = sorted(nfftdict.items(), key=lambda x: x[1])#按与当前framesize差值升序排列
framesize = int(sortlist[0][0])#取最接近当前framesize的那个2的正整数次方值为新的framesize
 
NFFT = framesize #NFFT必须与时域的点数framsize相等，即不补零的FFT
overlapSize = 1.0/3 * framesize #重叠部分采样点数overlapSize约为每帧点数的1/3~1/2
overlapSize = int(round(overlapSize))#取整
spectrum,freqs,ts,fig = plt.specgram(x,NFFT = NFFT,Fs =16000,window=np.hanning(M = framesize),noverlap=overlapSize,mode='default',scale_by_freq=True,sides='default',scale='dB',xextent=None)#绘制频谱图         
plt.ylabel('Frequency')
plt.xlabel('Time(s)')
plt.title('Spectrogram')
plt.show()

'''
'''
print("16kTO48k")
import os
import numpy as np
f = open("vad.wav",'rb')
f.seek(0)
f.read(44)
data = np.fromfile(f, dtype=np.int16)
#升采样
DATA=np.concatenate((data,data,data),axis=0)
DATA=DATA.reshape((3,len(data)))
DATA=DATA.reshape((-1),order='F')
print(DATA.shape)
print(DATA[0:30])
DATA.tofile("48k.pcm")
'''
import wave
import os

'''
f = open('output.pcm','rb')
str_data  = f.read()
wave_out=wave.open('123.wav','wb')
wave_out.setnchannels(1)
wave_out.setsampwidth(2)
wave_out.setframerate(48000)
wave_out.writeframes(str_data)
'''
'''
#str_data = f.readframes(nframes)
#wave_data = np.fromstring(str_data, dtype=np.short)
import matplotlib.pyplot as plt


fs,x=wav.read("vad.wav")
framelength = 0.040 #帧长20~30ms
framesize = framelength*16000 #每帧点数 N = t*fs,通常情况下值为256或512,要与NFFT相等\
                                    #而NFFT最好取2的整数次方,即framesize最好取的整数次方
#找到与当前framesize最接近的2的正整数次方
nfftdict = {}
lists = [32,64,128,256,512,1024]
for i in lists:
    nfftdict[i] = abs(framesize - i)
sortlist = sorted(nfftdict.items(), key=lambda x: x[1])#按与当前framesize差值升序排列
framesize = int(sortlist[0][0])#取最接近当前framesize的那个2的正整数次方值为新的framesize
 
NFFT = framesize #NFFT必须与时域的点数framsize相等，即不补零的FFT
overlapSize = 1.0/3 * framesize #重叠部分采样点数overlapSize约为每帧点数的1/3~1/2
overlapSize = int(round(overlapSize))#取整
plt.subplot(211)
spectrum,freqs,ts,fig = plt.specgram(x,NFFT = NFFT,Fs =16000,window=np.hanning(M = framesize),noverlap=overlapSize,mode='default',scale_by_freq=True,sides='default',scale='dB',xextent=None)#绘制频谱图         

plt.ylabel('Frequency')
plt.xlabel('Time(s)')
plt.title('Before Noise Suppression')



f=open('after_noise_reduction_48k.pcm','rb')
data=np.fromfile(f,dtype=np.int16)
data.shape=-1,3
data=data.T
DATA=data[0]
framelength = 0.040 #帧长20~30ms
print(DATA.shape)
framesize = framelength*16000 #每帧点数 N = t*fs,通常情况下值为256或512,要与NFFT相等\
                                    #而NFFT最好取2的整数次方,即framesize最好取的整数次方
nfftdict = {}
lists = [32,64,128,256,512,1024]
for i in lists:
    nfftdict[i] = abs(framesize - i)
sortlist = sorted(nfftdict.items(), key=lambda x: x[1])#按与当前framesize差值升序排列
framesize = int(sortlist[0][0])#取最接近当前framesize的那个2的正整数次方值为新的framesize
NFFT = framesize #NFFT必须与时域的点数framsize相等，即不补零的FFT 
overlapSize = 1.0/3 * framesize #重叠部分采样点数overlapSize约为每帧点数的1/3~1/2
overlapSize = int(round(overlapSize))#取整
plt.subplot(212)
spectrum,freqs,ts,fig = plt.specgram(DATA,NFFT = NFFT,Fs =16000,window=np.hanning(M = framesize),noverlap=overlapSize,mode='default',scale_by_freq=True,sides='default',scale='dB',xextent=None)#绘制频谱图         
plt.ylabel('Frequency')
plt.xlabel('Time(s)')
plt.title('After Noise Suppression')
plt.subplots_adjust(wspace =0, hspace =0.5)
plt.show()
'''
def AudPreEmphasize(signal):
    pre_emphasis = 0.93 #预加重系数 通常 0.9 < pre_emphasis < 1.0
    #Fs,signal=wav.read("202 (2).wav")
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    return emphasized_signal
    #emphasized_signal.dtype=np.int16
import matplotlib.pyplot as plt
f=open('after_noise_reduction_48k.pcm','rb')
data=np.fromfile(f,dtype=np.int16)
data.shape=-1,3
data=data.T
DATA=data[0]
framelength = 0.040 #帧长20~30ms
print(DATA.shape)
framesize = framelength*16000 #每帧点数 N = t*fs,通常情况下值为256或512,要与NFFT相等\
                                    #而NFFT最好取2的整数次方,即framesize最好取的整数次方
nfftdict = {}
lists = [32,64,128,256,512,1024]
for i in lists:
    nfftdict[i] = abs(framesize - i)
sortlist = sorted(nfftdict.items(), key=lambda x: x[1])#按与当前framesize差值升序排列
framesize = int(sortlist[0][0])#取最接近当前framesize的那个2的正整数次方值为新的framesize
NFFT = framesize #NFFT必须与时域的点数framsize相等，即不补零的FFT 
overlapSize = 1.0/3 * framesize #重叠部分采样点数overlapSize约为每帧点数的1/3~1/2
overlapSize = int(round(overlapSize))#取整
plt.subplot(211)
spectrum,freqs,ts,fig = plt.specgram(DATA,NFFT = NFFT,Fs =16000,window=np.hanning(M = framesize),noverlap=overlapSize,mode='default',scale_by_freq=True,sides='default',scale='dB',xextent=None)#绘制频谱图         
plt.ylabel('Frequency')
plt.xlabel('Time(s)')
plt.title('Before Pre-emphasis')

DATA2=AudPreEmphasize(DATA)
plt.subplot(212)
spectrum,freqs,ts,fig = plt.specgram(DATA2,NFFT = NFFT,Fs =16000,window=np.hanning(M = framesize),noverlap=overlapSize,mode='default',scale_by_freq=True,sides='default',scale='dB',xextent=None)#绘制频谱图         
plt.ylabel('Frequency')
plt.xlabel('Time(s)')
plt.title('After Pre-emphasis')
plt.subplots_adjust(wspace =0, hspace =0.5)
plt.show()
