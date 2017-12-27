# 垃圾短信分类

## 任务目标

实现一个垃圾短信识别系统，在给定的数据集上验证效果

## 数据介绍

带标签数据80W条，其中垃圾短信(label=1)8W条，正常短信(label=0)72W条

不带标签数据20W行(有部分空行)

## 思路

数据加载 ——> 清理文本[去除标点符号等]，中文分词[停用词] ——> 转换为矩阵[0/1或者TF-IDF] ——> 模型训练[调整参数] ——> 训练结果评估



## 思考

代价不同：将正常短信错分为垃圾短信的危害比将垃圾短信错分为正常短信的危害更大。

对于垃圾短信识别来说，precision更重要(判定为垃圾短信的一定是垃圾短信)


## 提升空间

 - SVM中惩戒系数和kernal的选择
 - 分词结果用O/1或者是TF-IDF表示
 - KNN中邻居数目的选择

---

![模型结果对比](https://github.com/CuiCh/Spam_Message_Classification/blob/master/result.jpg)


### 2017/12/22

代码说明：

- data_preprocessing.ipynb

数据预处理部分，包括加载数据，获得短信的向量表示并保存到矩阵文件

- classifier.ipynb

完成不同模型下的结果对比

注：直接加载矩阵文件，5W条短信会有3W多维度，出现Memory Error,尝试用PCA降维，情况有所缓解。

但尝试对10W条短信构建矩阵进行PCA降维时，会出现memory Error, 下一步考虑增量学习

目前采用的数据量为1W条短信，80%用作训练

从结果来看，结合F_score和precision,在不调整参数的情况下，SVM-linear和伯努利朴素贝叶斯的结果较好。


TODO：
- 增量学习



### 2017/12/02 

从网络上下载某个停用词列表（data/allstopwords.txt,共1893个，包括数字，标点符号，常见中文词汇），
去除停用词后分类精度反而下降，猜想有些停用词在标识垃圾短信方面还是有作用的，
如"!"在垃圾短信中出现的频次远高于正常短信。
用目前的短信文本构建自己的停用词列表，代码为ConstructStopwordsList.py，运行一次构建停用词列表，存储为data/stopwords.txt

评估每个函数运行时间

TODO：

- 测试整体样本上的时间
- 调整SVM的参数观察对结果的影响



### 2017/11/28 

基本流程跑通，使用SVM分类器进行试验

TODO：

- 测试整体样本上的时间
- 去除停用词和标点符号来减小向量空间
- 调整SVM的参数观察对结果的影响