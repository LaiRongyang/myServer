import numpy as np
#from keras.utils import np_utils
import os

from ML_Model import SVM_Model, MLP_Model


from Utils import load_model, Radar

import Opensmile_Feature as of
import Librosa_Feature as lf
from Config import Config

'''
Train(): 训练模型

输入:
	model_name: 模型名称（SVM / MLP / LSTM）
	save_model_name: 保存模型的文件名
    if_load: 是否加载已有特征（True / False）
    feature_method: 提取特征的方法（'o': Opensmile / 'l': librosa）
输出：
	model: 训练好的模型
'''


'''
Predict(): 预测音频情感
输入:
	model: 已加载或训练的模型
	model_name: 模型名称
	file_path: 要预测的文件路径
    feature_method: 提取特征的方法（'o': Opensmile / 'l': librosa）
输出：
    预测结果和置信概率
'''
def Predict(model, model_name: str, file_path: str, feature_method: str = 'Opensmile'):
    
    file_path = os.path.dirname(os.path.abspath(__file__)) + '/' + file_path

    if(feature_method == 'o'):
        # 一个玄学 bug 的暂时性解决方案
        of.get_data(file_path, Config.PREDICT_FEATURE_PATH_OPENSMILE, train = False)
        test_feature = of.load_feature(Config.PREDICT_FEATURE_PATH_OPENSMILE, train = False)
    elif(feature_method == 'l'):
        test_feature = lf.get_data(file_path, Config.PREDICT_FEATURE_PATH_LIBROSA, train = False)       
    result = model.predict(test_feature)

    result_prob = model.predict_proba(test_feature)[0]
    print('Recogntion: ', Config.CLASS_LABELS[int(result)])
    print('Probability: ', result_prob)
    
    #雷达图显示
    #Radar(result_prob)
    #return Config.CLASS_LABELS[int(result)]
    return (result_prob)



# model = Train(model_name = "lstm", save_model_name = "LSTM_LIBROSA", if_load = True, feature_method = 'l')
# 加载模型
# model = load_model(load_model_name = "LSTM_LIBROSA", model_name = "lstm")
# Predict(model, model_name = "lstm", file_path = "Test/247-fear-wangzhe.wav", feature_method = 'l')
