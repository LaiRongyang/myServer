#model_4_24_4c_2 准确率还可以
#normal

import numpy as np
import tensorflow as tf
from data.train_data import process_save_wav_sample
from data.train_data import process_save_wav_sample_x
from data.train_data import AudPreEmphasize
import wave
import os
import scipy.io.wavfile as wav
from data.example import myvad


IS_training=False
tf_x = tf.placeholder(tf.float32, [None, 400, 200])
image = tf.reshape(tf_x, [-1, 400, 200, 1])
tf_y = tf.placeholder(tf.float32, [None,4])

conv_time = tf.layers.conv2d(inputs=image, filters=8, kernel_size=[2, 8], strides=1, padding='same', activation=None,name='conv_time', kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-4))
b1=tf.layers.batch_normalization(conv_time, training=IS_training,name='b1')
bn_time = tf.nn.relu(b1)
conv_freq = tf.layers.conv2d(inputs=image, filters=8, kernel_size=[10, 2], strides=1, padding='same', activation=None,name='conv_freq',kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-4))
b2=tf.layers.batch_normalization(conv_freq, training=IS_training,name='b2')
bn_freq = tf.nn.relu(b2)
time_freq = tf.concat([bn_time, bn_freq], 3)

pool_time_freq = tf.layers.max_pooling2d(time_freq, pool_size=(2, 2), strides=2)

conv1 = tf.layers.conv2d(inputs=pool_time_freq, filters=32, kernel_size=(3, 3), strides=1, padding='same',
                         activation=None,name='conv1',kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-4))
b3=tf.layers.batch_normalization(conv1, training=IS_training,name='b3')

pool1 = tf.layers.max_pooling2d(tf.nn.relu(b3), pool_size=(2, 2),
                                strides=2)

conv2 = tf.layers.conv2d(inputs=pool1, filters=48, kernel_size=(3, 3), strides=1, padding='same', activation=None,name='conv2',kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-4))
b4=tf.layers.batch_normalization(conv2, training=IS_training,name='b4')
pool2 = tf.layers.max_pooling2d(tf.nn.relu(b4), pool_size=(2, 2),
                                strides=2)

conv3 = tf.layers.conv2d(inputs=pool2, filters=64, kernel_size=(3, 3), strides=1, padding='same', activation=None,name='conv3',kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-4))
b5=tf.layers.batch_normalization(conv3, training=IS_training,name='b5')
acti1 = tf.nn.relu(b5)

conv4 = tf.layers.conv2d(inputs=acti1, filters=80, kernel_size=(3, 3), strides=1, padding='same', activation=None,name='conv4',kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-4))
b6=tf.layers.batch_normalization(conv4, training=IS_training,name='b6')
acti2 = tf.nn.relu(b6)

top_down_attention = tf.layers.conv2d(inputs=acti2, filters=4, kernel_size=(1, 1), padding='valid', activation=None,name='top_down_attention',kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-4))
bottom_up_attention = tf.layers.conv2d(inputs=acti2, filters=1, kernel_size=(1, 1), padding='valid', activation=None,name='bottom_up_attention',kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-4))

xxx = tf.reshape(top_down_attention * bottom_up_attention, shape=(-1, 1250, 4))

output = tf.reduce_mean(xxx, axis=1)

prediction=tf.argmax(tf.nn.softmax(output),1)
predict=tf.nn.softmax(output)

l2_loss = tf.losses.get_regularization_loss()
loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)+l2_loss

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op1 = tf.train.MomentumOptimizer(0.05, 0.9,use_nesterov=True).minimize(loss)
    train_op2 = tf.train.MomentumOptimizer(0.005, 0.9, use_nesterov=True).minimize(loss)
    train_op3 = tf.train.MomentumOptimizer(0.0005, 0.9, use_nesterov=True).minimize(loss)
    train_op4 = tf.train.MomentumOptimizer(0.00005, 0.9, use_nesterov=True).minimize(loss)


sess = tf.Session()
init_op = tf.group(   tf.global_variables_initializer(),tf.local_variables_initializer())  # the local var is for accuracy_op
sess.run(init_op)



def init():
    save_dir = 'train_model_4_24_4c_2/mymodel.ckpt'
    saver=tf.train.Saver(max_to_keep=1,var_list=tf.global_variables())
    saver.restore(sess, save_dir)
init()




        


import socket
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
import threading
import time

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

#接收收据
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
            paths=myvad(fname,3)
            os.remove(fname)
            print("输出后")
            if len(paths)>0:
                frames=[]
                frames=np.array(frames)
                for file in paths:
                    print("切出一个文件")
                    frames=np.concatenate((frames,wav.read(file)[1]),axis=0)
                print(frames.shape)
                frames=AudPreEmphasize(frames)
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



















