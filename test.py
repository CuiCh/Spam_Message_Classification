
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer  
import sklearn
import jieba
from scipy import io

docs = ["为了扩展互联网汽车的数字服务今天","今天举行大促的销","你永远不会知道的！","标点那符的号 ， 。 ？","带给的我们大常州一场壮观的视觉盛宴","虎子一脸的悲催：遇到股票上涨"]


class MessageCountVectorizer(sklearn.feature_extraction.text.CountVectorizer):
    def build_analyzer(self):
        def analyzer(doc):
            words = jieba.cut(doc)
            return words
        return analyzer

# 去除出现在90%文本中的词，去除出现次数不到两次的词
vect = 	MessageCountVectorizer(min_df=2,max_df=0.9)
vect_result=vect.fit_transform(docs)

words = vect.get_feature_names()
print(words)
print(len(words))
print(vect_result.toarray())
io.mmwrite("data/word_vector.mtx",vect_result)