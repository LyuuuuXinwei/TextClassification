# TextClassification
获取搜狗实验室的原始搜狐新闻xml数据，转换清理，从url提取类别标签，得到11个类别的约22000条新闻文本。将文本划分为训练验证测试集，分别使用TF-IDF+传统的统计学习模型(贝叶斯/svm/感知机)与word2vec+深度学习模型(cnn/lstm/fasttext)进行分类训练，对预测结果投票融合，测试集准确率0.866。  
