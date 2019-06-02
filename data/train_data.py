#创建训练集
import os
import numpy as np
import python_speech_features as ps
import scipy.io.wavfile as wav
#################################################################################

def segment_axis(a, length, overlap=0, axis=None, end='cut', endvalue=0):
    if axis is None:
        a = np.ravel(a) # may copy
        axis = 0
        # 多维数组降为一维，np.ravel()返回的是视图，修改时会影响原始矩阵
    l = a.shape[axis]
    if overlap >= length:
        raise( ValueError, "frames cannot overlap by more than 100%")
    if overlap < 0 or length <= 0:
        raise (ValueError, "overlap must be nonnegative and length must "\
                          "be positive")
    if l < length or (l-length) % (length-overlap):
        if l>length:
            roundup = length + (1+(l-length)//(length-overlap))*(length-overlap)
            rounddown = length + ((l-length)//(length-overlap))*(length-overlap)
        else:
            roundup = length
            rounddown = 0
        assert rounddown < l < roundup
        assert roundup == rounddown + (length-overlap) \
               or (roundup == length and rounddown == 0)
        a = a.swapaxes(-1,axis)

        if end == 'cut':
            a = a[..., :rounddown]
        elif end in ['pad','wrap']: # copying will be necessary
            s = list(a.shape)
            s[-1] = roundup
            b = np.empty(s,dtype=a.dtype)
            b[..., :l] = a
            if end == 'pad':
                b[..., l:] = endvalue
            elif end == 'wrap':
                b[..., l:] = a[..., :roundup-l]
            a = b

        a = a.swapaxes(-1,axis)


    l = a.shape[axis]
    if l == 0:
        raise (ValueError, \
              "Not enough data points to segment array in 'cut' mode; "\
              "try 'pad' or 'wrap'")
    assert l >= length
    assert (l-length) % (length-overlap) == 0
    n = 1 + (l-length) // (length-overlap)
    s = a.strides[axis]
    newshape = a.shape[:axis] + (n,length) + a.shape[axis+1:]
    newstrides = a.strides[:axis] + ((length-overlap)*s,s) + a.strides[axis+1:]

    try:
        return np.ndarray.__new__(np.ndarray, strides=newstrides,
                                  shape=newshape, buffer=a, dtype=a.dtype)
    except TypeError:
        warnings.warn("Problem with ndarray creation forces copy.")
        a = a.copy()
        # Shape doesn't change but strides does
        newstrides = a.strides[:axis] + ((length-overlap)*s,s) \
                     + a.strides[axis+1:]
        return np.ndarray.__new__(np.ndarray, strides=newstrides,
                                  shape=newshape, buffer=a, dtype=a.dtype)

def process_save_wav_sample(path):
    [Fs, x] = wav.read(path) # 读取wav的原始信号
    if(len(x)>32000):
        utterances_train = segment_axis(x, 32000, Fs) # 训练shift 1s
        utterances_test = segment_axis(x, 32000, int(Fs*0.4)) # 测试shift 0.4s
    else:
        utterances_train = [x]
        utterances_test = [x]
    # utterances_train = frame_seg(x.tolist(), 32000, Fs)
    # utterances_test = frame_seg(x.tolist(), 32000, int(Fs*0.4))
    X_train=[]
    for i, x_train in enumerate(utterances_train):
        x_train = save_feature_spectrogram(x_train, Fs)
        X_train.append(x_train)
    return X_train
def process_save_wav_sample_x(data):
    #[Fs, x] = wav.read(path) # 读取wav的原始信号
    Fs=16000
    x=data	
    if(len(x)>32000):
        utterances_train = segment_axis(x, 32000, Fs) # 训练shift 1s
        utterances_test = segment_axis(x, 32000, int(Fs*0.4)) # 测试shift 0.4s
    else:
        utterances_train = [x]
        utterances_test = [x]
    # utterances_train = frame_seg(x.tolist(), 32000, Fs)
    # utterances_test = frame_seg(x.tolist(), 32000, int(Fs*0.4))
    X_train=[]
    for i, x_train in enumerate(utterances_train):
        x_train = save_feature_spectrogram(x_train, Fs)
        X_train.append(x_train)
    return X_train
def save_feature_spectrogram(x, Fs):
    win = int(Fs*0.04)
    step = int(Fs*0.01)
    frames =   ps.sigproc.framesig(x, frame_len=win, frame_step=step, winfunc=np.hanning)
    x = ps.sigproc.logpowspec(frames, 1600)
    x = x[:,:400]
    x = np.flipud(np.transpose(x))  # arr.shape == 400x200, 400表示0-4khz，200表示2s
    x = np.pad(x, ((0, 0), (0, 200 - len(x[0]))), mode='constant', constant_values=.0)
    return x
#预加重
def AudPreEmphasize(signal):
    pre_emphasis = 0.97 #预加重系数 通常 0.9 < pre_emphasis < 1.0
    #Fs,signal=wav.read("202 (2).wav")
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    return emphasized_signal
    #emphasized_signal.dtype=np.int16
count=0
def abc(path,name):
    global count
    f,x=wav.read(path)
    data=AudPreEmphasize(x)
    nx=process_save_wav_sample_x(data)
    for i in range(len(nx)):
        np.save(name+str(count),nx[i])
        count=count+1
    

    
###################################################################################################
'''
def generate_data():
    environment=0
    env=process_save_wav_sample("E:/alarmmmmmmmmmmm/self_data_4_14/self_data/environment/library.wav")
    for i in range(len(env)):
	np.save("environment"+str(environment),env[i])
	environment=environment+1

    speak=0
    import os
    path=os.listdir("E:/alarmmmmmmmmmmm/self_data_4_14/self_data/speak")
    int_path=["E:/alarmmmmmmmmmmm/self_data_4_14/self_data/speak/"+i for i in path]
    for i in int_path:
	nx=process_save_wav_sample(i)
	for j in nx:
		np.save("speak"+str(speak),j)
		speak=speak+1

    fear=0
    fe=process_save_wav_sample("E:/alarmmmmmmmmmmm/self_data_4_14/self_data/fear/bbbbb.wav")
    for i in range(len(fe)):
	np.save("fear"+str(fear),fe[i])
	fear=fear+1

    angry=0
    path=os.listdir("E:/alarmmmmmmmmmmm/self_data_4_14/self_data/angry")
    int_path=["E:/alarmmmmmmmmmmm/self_data_4_14/self_data/angry/"+ i for i in path]
    for i in  int_path:
	nx=process_save_wav_sample(i)
	for j in nx:
		np.save("angry"+str(angry),j)
		angry=angry+1

   ''' '''
>>> train_npy_path=os.listdir("E:/alarmmmmmmmmmmm/self_data_4_14/self_data_npy")
>>> path=["E:/alarmmmmmmmmmmm/self_data_4_14/self_data_npy/"+i for i in train_npy_path]
>>> len(path)
1216
>>> path=np.array(path)
>>> path=path.reshape(4,-1)
>>> path.shape
(4, 304)
>>> train_npy_path=os.listdir("E:/alarmmmmmmmmmmm/self_data_4_14/self_data_npy")
>>> path=["E:/alarmmmmmmmmmmm/self_data_4_14/self_data_npy/"+i for i in train_npy_path]
>>> len(path)
1216
>>> path=np.array(path)
>>> path=path.reshape((4,-1))
>>> patj.shape
Traceback (most recent call last):
  File "<pyshell#55>", line 1, in <module>
    patj.shape
NameError: name 'patj' is not defined
>>> path.shape
(4, 304)
>>> path_inre=[path[:,i] for i in range(304)]
>>> path_inre=np.array(path_inre)
>>> path_inre=path_inre.reshape(-1)
>>> path_inre
    dtype='<U66')
>>> fear=0
>>> for i in path_inre:
	if"fear" in i:
		fear=fear+1

		
>>> fear
192
>>> len(env)
518
>>> env=0
>>> train_data=[]
>>> train_label=[]
>>> for i in path_inre:
	x=np.load(i)
	train_data.append(x)
	if "environment" in i:
		train_label.append([1,0,0,0])
	elif "speak" in i:
		train_label.append([0,1,0,0])
	elif "fear" in i:
		train_label.append([0,0,1,0])
	elif "angry" in i:
		train_label.append([0,0,0,1])
	else:print("error")

	
>>> len(train_data)
1216
>>> len(train_label)
1216
>>> train_data=np.array(train_data)
>>> train_label=np.array(train_label)
>>> np.save("train_data_1216",train_data)
>>> np.save("train_label_1216",train_label)
>>> 
  )#保存特征集 和标签集'''
####################################################################
