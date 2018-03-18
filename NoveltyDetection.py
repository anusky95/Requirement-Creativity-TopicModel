import sys
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import OneClassSVM

df = pd.read_csv('correctedNovelty.csv',sep=',',names=['rate','feature'])

print("Number of novel requirements", len(df[df.rate==-1]))
print("Number of non-novel requirements", len(df[df.rate==1]))
df_x = df["feature"]
df_y = df["rate"]
cv = CountVectorizer()
x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.2,random_state=4)

print("Length of training data:",len(x_train))
print("Length of test data:",len(x_test))

print("Train Data novelty classification stats:")
print("Length of not novel requirements in training dataset",len(x_train[df.rate==1]))
print("Length of novel requirements in training dataset",len(x_train[df.rate==-1]))

nonNovelTrainSet = x_train[df.rate==1]
NovelTrainSet = x_train[df.rate==-1]

cv1 = CountVectorizer()
x_traincv1 = cv1.fit_transform(x_train.values.astype('U'))
x_testcv = cv1.transform(x_test.values.astype('U'))
nonNovelTrainSet_cv = cv1.transform(nonNovelTrainSet.values.astype('U'))
NovelTrainSet_cv = cv1.transform(NovelTrainSet.values.astype('U'))

clf = IsolationForest(max_samples="auto")
clf.fit(nonNovelTrainSet_cv)
y_pred_test = clf.predict(x_testcv)
arr = np.array(y_test)


result_IsolationForest = classification_report(arr,y_pred_test)
print("Classification report for Isolation Forest:")
print(result_IsolationForest)


clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(NovelTrainSet_cv)
ytest = clf.fit_predict(x_testcv)

result_LocalOutlierFactor = classification_report(arr,ytest)
print("Classification report for Local Outlier Factor:")
print(result_LocalOutlierFactor)


clf = svm.OneClassSVM(nu=0.1,kernel="rbf",gamma=0.1)
clf.fit(nonNovelTrainSet_cv)
svmoneclass = clf.predict(x_testcv)
#print(svmoneclass)
print("Classification report for SVM One class")
result = classification_report(arr,svmoneclass)
print(result)




