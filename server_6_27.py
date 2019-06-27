import   myser

import threading
import time

import numpy as np
import tensorflow as tf
from data.train_data import process_save_wav_sample
from data.train_data import process_save_wav_sample_x
from data.train_data import AudPreEmphasize
import wave
import os
import scipy.io.wavfile as wav
from data.example import myvad
import socket
import  sox
import threading
import time
cbn = sox.Combiner()#合并音频文件
#myser.get_emotion(test.wav)

s = socket.socket() #Create a socket object
bsize = s.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
print(bsize)
host = socket.gethostname() #Get the local machine name
port = 12397 # Reserve a port for your service
s.bind(("",port)) #Bind to the port
s.listen(5) #Wait for the client connection
data=[]

'''
结果

[[0.02977182 0.70166206 0.23576824 0.03279793]]
[[2.7817843e-02 8.8926041e-01 8.2342632e-02 5.7907921e-04]]
[[0.07180029 0.8360131  0.08997783 0.0022088 ]]
[[9.8325349e-03 9.2863399e-01 6.1283104e-02 2.5037417e-04]]

'''


# 为线程定义一个函数
def print_time( threadName, delay):
   count = 0
   while count < 5:
      time.sleep(delay)
      count += 1
      print ("%s: %s" % ( threadName, time.ctime(time.time()) ))

    
def ER(x):
    n=process_save_wav_sample_x(x)
    n=np.array(n)
    out=sess.run(output,feed_dict={tf_x:n})
    print(out)
    return out


#处理数据，并发送结果
def process_and_send(client,address,data):#socket
    print("收到shuju",len(data))
    result=np.round(ER(data),decimals=3)
    for i in result:
        #client.send(bytes(str(i)+'\n',encoding='utf8'))
        #只发送结果
        j=i
        j=j.tolist()
        index=j.index(max(j))
        print("结果：",index)
        client.send(bytes(str(index)+'\n',encoding='utf8'))

#接收收据 当接收超过32000帧语音 
def received_audio_thread(client,address):#socket    
    data=np.array([])
    fname=str(time.time())+".wav"
    f=wave.open(fname,"wb")  
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(16000)
    print("创建文件")
    if(f== None):
        print('error')
    while(True):
        
        recv_data=client.recv(8000)#经过测试，发现通道断开或者close之后，就会一直收到空字符串。 而不是所谓的-1 或者报异常。
        if(len(recv_data)==0):
            f.close()
            client.close()
            os.remove(fname)
            print("客户端关闭连接")
            return
            break
        f.writeframes(recv_data)
        print("写一次",len(recv_data))
        data=np.concatenate((np.frombuffer(recv_data,dtype="<i2"),data),axis=0)
        if(len(data))>=32000:
            
            data=np.array([])
            f.close()
            print("输出前")
            paths=myvad(fname,3)#这个是py 语音活动检测  ，输入wav返回一段语音中的有效部分 返回的是有效部分 的wav文件列表
            os.remove(fname)
            print("输出后")
            if len(paths)>0:
                #frames=[]
                #frames=np.array(frames)   
                #for file in paths:    
                    #print("切出一个文件")
                    #frames=np.concatenate((frames,wav.read(file)[1]),axis=0)
                #print(frames.shape)
                #frames=AudPreEmphasize(frames)
                outfile_name=str(time.time())+".wav"
                cbn.build(paths,outfile_name,'concatenate')#合并文件
                #开一个线程，
                thread=threading.Thread(target=process_and_send,args=(client,address,frames))
                thread.start()
            else:
                print("发送结果0：")
                client.send(bytes(str(0)+'\n',encoding='utf8'))
                
            fname=str(time.time())+".wav"    
            f=wave.open(fname,"wb")
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(16000)
while True:
    c,addr = s.accept() #Establish a connection with the client
    print( "Got connection from", addr)
    #_thread.start_new_thread( print_time, ("Thread-1", 2, ) )
    try:
        thread=threading.Thread(target=received_audio_thread,args=(c,addr))
        thread.start()
        #_thread.start_new_thread(received_audio_thread,(c))
    except:
        print("error")



















