#encoding:utf-8
#交叉验证
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

'''测试集
1. 过拟合： 模型容易拟合训练数据，但是在新的数据上预测效果很差，因此不能在训练集上进行验证。
2. 为了防止过拟合，一般的做法是拿出一部分训练数据作为测试集。
3. sklearn中训练集和测试集可以通过·train_test_split·函数得到。
'''
#加载iris数据
iris=datasets.load_iris()
ds=iris.data.shape
ts=iris.target.shape
print(ds,ts)
#采样训练数据和测试数据
X_train,X_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.4,random_state=0)

clf=svm.SVC(kernel='linear',C=1.).fit(X_train,y_train)
train_score=clf.score(X_train,y_train)
print(train_score)
score=clf.score(X_test,y_test)
print(score)


'''
交叉验证
1.  需要测试集，不需要验证集
2. k-fold交叉验证，训练数据分成K个小的集合，K fold中的每个小的集合的处理过程如下：
  * 模型使用K-1个folds小集合作为训练数据。
  * 训练的模型在剩下的数据上验证。
3. 耗费更多的计算，但是没有浪费太多数据。
4. 计算交叉验证的metrics可以使用cross_val_score方法来实现
5. cv参数是整数时，使用k-fold交叉验证，会得到k个得分; cv也可以使用交叉验证迭代器：ShuffleSplit 
6. 默认的得分计算是使用estimator的score方法，也可以使用`scoring`参数设置
'''
from sklearn.model_selection import cross_val_score
clf=svm.SVC(kernel='linear',C=1)
scores=cross_val_score(clf,iris.data,iris.target,cv=5,scoring='f1_macro')
print(scores)

#使用validation iterator
from sklearn.model_selection import ShuffleSplit
n_samples=iris.data.shape[0]
cv=ShuffleSplit(n_splits=3,test_size=0.3,random_state=0)
scores=cross_val_score(clf,iris.data,iris.target,cv=cv)
print(scores)

'''
数据转换（标准化等）
1. 由于在训练数据意外的数据上预测很重要，预处理（比如标准化，特征选择等）和
   相似数据转换(data transformation)应该从训练数据中学习并应用到测试数据的预测中。
2. 数据的预处理可以使用sklearn的·preprocessing·包里面的方法
3. 可以使用Pipeline组合多个estimators
'''
from sklearn import preprocessing
scalar=preprocessing.StandardScaler().fit(X_train)
X_train_transformed=scalar.transform(X_train)
clf=svm.SVC(C=1).fit(X_train_transformed,y_train)
X_test_transformed=scalar.transform(X_test)
scores=clf.score(X_test_transformed,y_test)
print(scores)

#使用Piplie组合
from sklearn.pipeline import make_pipeline
clf=make_pipeline(preprocessing.StandardScaler(),svm.SVC(C=1))
scores=cross_val_score(clf,iris.data,iris.target,cv=cv)
print(scores)

'''
使用交叉验证预测
1. 使用cross_val_predict方法实现交叉验证， 该函数与`cross_val_score`类似，但是返回的是预测结果。
'''
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
predicted=cross_val_predict(clf,iris.data,iris.target,cv=10)
scores=metrics.accuracy_score(iris.target,predicted)
print(scores)

'''
交叉验证迭代器：cross validation iterator
1. KFold 把样本分成K组
2. LeaverOneOut: 
'''