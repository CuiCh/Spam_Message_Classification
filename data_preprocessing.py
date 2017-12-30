
# coding: utf-8

# In[3]:

import jieba
import time
import pandas as pd
import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer  
from scipy import io
import json


# In[4]:

def logtime(func):
    """
    函数目的：测量函数运行时间 
    Parameter:
        func - 被测量的函数
    Return:
        wrapper - 被装饰之后的函数
    """
    def wrapper(*args,**kwargs):
        start = time.time()
        result = func(*args,**kwargs)
        end = time.time()
        print("完成函数{name}, 运行时间 {totaltime:.3f}s".format(name=func.__name__,totaltime=end-start))
        start = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(start))
        end = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(end))
        print("开始时间 : %s \n结束时间 : %s "%(start,end))
        return result
    return wrapper


# In[5]:

def load_data(rawdata,n):
    """
    Purpose:加载原始数据，处理并输出
    
    """
    alldata = pd.read_csv(rawdata,header=None)
    alldata.columns = ["label","content"]
    data = alldata.sample(n)
    content = data["content"]
    label=data["label"]
    return content,label


# In[6]:

class MessageCountVectorizer(sklearn.feature_extraction.text.CountVectorizer):
    def build_analyzer(self):
        def analyzer(doc):
            words = jieba.cut(doc)
            return words
        return analyzer


# In[7]:

@logtime
def vect_data(content,label):
    """
    函数说明：得到每个短信的内容和标签的向量表示，同时保存特征词
    Return:
        vect_result - 短信的向量表示
        label - 标签的向量表示
        words - 词汇表
    Modify:
        2017-12-22
    
    """
    vect = 	MessageCountVectorizer(max_df=0.9,min_df=2)
    vect_result=vect.fit_transform(content)
    io.mmwrite("data/content_vector.mtx",vect_result)
    
    label = label.tolist()
    with open('data/label_vector.json', 'w') as f:
        json.dump(label, f)
        
    words = vect.get_feature_names()
    print("使用了%d条短信,词汇表长度:%s"%(len(label),len(words)))
    with open('data/feature_words.json', 'w') as f:
        json.dump(words, f)


# In[16]:

def main():
    rawdata_path = "rawdata/traindata.csv"
    content,label=load_data(rawdata_path,n=10000)
    vect_data(content,label)


# In[17]:

if __name__ == "__main__":
    main()


# In[ ]:



