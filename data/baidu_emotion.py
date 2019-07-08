from aip import AipSpeech
from aip import AipNlp
'''暂时不考虑error的情况'''
'''文档地址http://ai.baidu.com/docs#/ASR-Online-Python-SDK/top
http://ai.baidu.com/docs#/NLP-Python-SDK/top'''

 
""" 你的 APPID AK SK """
get_text_client = AipSpeech('16461114', 'TirtQku2WIdr4Wtv47PIn1BG', 'QUYsEtPFLS3BpoOHf0FvG6mYG3PYIN7m')
get_emoion_client = AipNlp('16656595', 'VhRcQ1BUNeGXUvKLsVOQQCoK', 'jvXmGGKCfhmaI4Ne1Z6phtnEqq9LH9kC')


# 读取文件
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

#获取wav 文件的语义情感信息
'''返回字符串示例
neutral:0.997434,pessimistic:0.0023148,optimistic:0.000251054'''
'''标签顺序按概率顺序排列'''
def get_wav_emotion(wav_path): 
    # 识别本地文件
    text_result=get_text_client.asr(get_file_content(wav_path), 'wav', 16000, {'dev_pid': 1536,})
    #取返回结果的第一个
    try:
        text_result=text_result['result'][0]
        emotion_result=get_emoion_client.emotion(text_result)['items']
    
        return (emotion_result[0]['label']+':'+str(emotion_result[0]['prob'])+','+
                emotion_result[1]['label']+':'+str(emotion_result[1]['prob'])+','+
                emotion_result[2]['label']+':'+str(emotion_result[2]['prob']))
    except:
        return ('')

    
#print(get_wav_emotion('test.wav'))    
    
    
    

    
    
    
