import pandas as pd
import jieba

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
    stopwordinham=sorted(stopwordinham.items(),key=lambda item:item[1],reverse=True)
    stopwordinham = stopwordinham[:200]
    print(stopwordinham)
    stopwordinham = list(zip(*stopwordinham))[0]
    stopwordinham = set(stopwordinham)

    print("- - -- - - - - - -  -  - - -")
    stopwordinspam = {}
    for i in range(len(spam)):
        content = ham.iloc[i]
        segresult = set(jieba.cut(content))
        for word in segresult:
            if word in stopwordinspam:
                stopwordinspam[word]+=1
            else:
                stopwordinspam[word]=1
    stopwordinspam=sorted(stopwordinspam.items(),key=lambda item:item[1],reverse=True)
    stopwordinspam = stopwordinspam[:200]
    print(stopwordinspam)
    stopwordinspam=list(zip(*stopwordinspam))[0]
    stopwordinspam = set(stopwordinspam)

    
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