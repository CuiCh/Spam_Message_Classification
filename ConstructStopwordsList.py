import pandas as pd
import jieba
import time


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

@logtime
def construct_stop_words(data):
    """
    函数目的：通过现有的文本构建停用词列表
    Parameter:
        data - 文本数据
    Return:
        创建停用词文件
    """
    ham = data[data["label"]==0]["content"]
    spam = data[data["label"]==1]["content"]

    stopwordinham = {}
    for i in range(len(ham)):
        content = ham.iloc[i]
        segresult = set(jieba.cut(content))
        for word in segresult:
            if word in stopwordinham:
                stopwordinham[word]+=1
            else:
                stopwordinham[word]=1
    # 获取正常短信中前两百个频繁词
    stopwordinham=sorted(stopwordinham.items(),key=lambda item:item[1],reverse=True)
    stopwordinham = stopwordinham[:200]
    print(stopwordinham)
    stopwordinham = list(zip(*stopwordinham))[0]
    stopwordinham = set(stopwordinham)

    print("- - -- - - - - - -  -  - - -")

    stopwordinspam = {}
    for i in range(len(spam)):
        content = spam.iloc[i]
        segresult = set(jieba.cut(content))
        for word in segresult:
            if word in stopwordinspam:
                stopwordinspam[word]+=1
            else:
                stopwordinspam[word]=1

    # 获取垃圾短信中前两百个频繁词
    stopwordinspam=sorted(stopwordinspam.items(),key=lambda item:item[1],reverse=True)
    stopwordinspam = stopwordinspam[:200]
    print(stopwordinspam)
    stopwordinspam=list(zip(*stopwordinspam))[0]
    stopwordinspam = set(stopwordinspam)

    # 同时出现在垃圾短信和正常短信中的频繁词作为停用词
    stopwords = stopwordinspam & stopwordinham
    print(stopwords)
    print(len(stopwords))
    with open("data/stopwords.txt","w",encoding="utf-8") as f:
        for stopword in stopwords:
            f.write(stopword+'\n')

def main():
    filepath = "data/traindata.csv"
    data = pd.read_csv(filepath,header=None)
    data.columns=["label","content"]
    # 选择全部数据
    data = data.iloc[:800000]
    construct_stop_words(data)

if __name__ == '__main__':
    main()