import argparse
from SER import Train, Predict
from Utils import load_model
model = load_model(load_model_name = 'SVM_LIBROSA', model_name = 'svm')



'''返回 CLASS_LABELS = ("angry", "fear", "happy", "neutral", "sad", "surprise")

 '''

def get_emotion(path):
    print(Predict(model, model_name = 'svm', file_path = path, feature_method = 'l'))
    
