
# coding: utf-8

# In[76]:

import sys
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.naive_bayes import MultinomialNB
#from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


df = pd.read_csv('C:/Users/Anupama/Documents/4thsem/THESIS/OutputThreshold/Requirement_model3_3.csv',sep=',')
print("Data basic description:")
print(df.describe())
print("Data column")
print(df.columns)


# In[77]:

print("Number of novel requirements:", len(df[df.novelty==-1]))
print("Number of non-novel requirements:", len(df[df.novelty==1]))
print("Number of novel clarity scores:", len(df[df.clarity==-1]))
print("Number of non-novel clarity scores:", len(df[df.clarity==1]))
print("Number of novel creativity scores:", len(df[df.creativity_score==-1]))
print("Number of non-novel creativity scores:", len(df[df.creativity_score==1]))
print("Number of novel usefulness scores:", len(df[df.usefulness==-1]))
print("Number of non-novel usefulness scores:", len(df[df.usefulness==1]))

df_requirement = df["requirement"]
df_novelty = df["novelty"]
df_creativity = df["creativity_score"]
df_clarity = df["clarity"]
df_usefulness = df["usefulness"]
df_reqOriginal = df["originalRequirement"]


# In[78]:

#### Splitting data to training and testing dataset
requirement_train, requirement_test, novelty_train, novelty_test = train_test_split(df_requirement,df_novelty,test_size=0.2,random_state=4)
requirement_train, requirement_test, clarity_train, clarity_test = train_test_split(df_requirement,df_clarity,test_size=0.2,random_state=4)
requirement_train, requirement_test, creativity_train, creativity_test = train_test_split(df_requirement,df_creativity,test_size=0.2,random_state=4)
requirement_train, requirement_test, usefulness_train, usefulness_test = train_test_split(df_requirement,df_usefulness,test_size=0.2,random_state=4)

print("################################")

print("Length of training data:",len(requirement_train))
print("Length of test data:",len(requirement_test))
print("################################")

print("Train data stats: Total: ",len(requirement_train))
print("----------------------")

print("1. Regular novelty values:",len(requirement_train[df.novelty==1]))
print("1. Novel values:",len(requirement_train[df.novelty==-1]))

print("2. Regular clarity values:",len(requirement_train[df.clarity==1]))
print("2. Novel clarity values:",len(requirement_train[df.clarity==-1]))

print("3. Regular creativity scores:",len(requirement_train[df.creativity_score==1]))
print("3. Novel creativity scores:",len(requirement_train[df.creativity_score==-1]))

print("4. Regular usefulness values:",len(requirement_train[df.usefulness==1]))
print("4. Novel usefulness values:",len(requirement_train[df.usefulness==-1]))


print("################################")
print("----------------------")

print("Test data stats: Total: ",len(requirement_test))

print("1. Regular novelty values:",len(requirement_test[df.novelty==1]))
print("1. Novel values:",len(requirement_test[df.novelty==-1]))

print("2. Regular clarity values:",len(requirement_test[df.clarity==1]))
print("2. Novel clarity values:",len(requirement_test[df.clarity==-1]))

print("3. Regular creativity scores:",len(requirement_test[df.creativity_score==1]))
print("3. Novel creativity scores:",len(requirement_test[df.creativity_score==-1]))

print("4. Regular usefulness values:",len(requirement_test[df.usefulness==1]))
print("4. Novel usefulness values:",len(requirement_test[df.usefulness==-1]))



print(df_requirement.value_counts())



# In[79]:

cv = CountVectorizer()
regularNoveltySet = requirement_train[df.novelty==1]
novelNovelSet = requirement_train[df.novelty==-1]
regularClaritySet = requirement_train[df.clarity==1]
novelClaritySet = requirement_train[df.clarity==-1]
regularCreativeSet = requirement_train[df.creativity_score==1]
novelCreativeSet = requirement_train[df.creativity_score==-1]
regularUsefulSet = requirement_train[df.usefulness==1]
novelUsefulSet = requirement_train[df.usefulness==-1]


requirementTrain = cv.fit_transform(requirement_train.values.astype('U'))
requirementTest = cv.fit_transform(requirement_test.values.astype('U'))

regularNoveltySet_converted = cv.transform(regularNoveltySet.values.astype('U'))
NovelNovelSet_converted = cv.transform(novelNovelSet.values.astype('U')) 

regularClaritySet_converted = cv.transform(regularClaritySet.values.astype('U'))
NovelClaritySet_converted = cv.transform(novelClaritySet.values.astype('U')) 

regularCreativeSet_converted = cv.transform(regularCreativeSet.values.astype('U'))
NovelCreativeSet_converted = cv.transform(novelCreativeSet.values.astype('U')) 

regularUsefulSet_converted = cv.transform(regularUsefulSet.values.astype('U'))
NovelUsefulSet_converted = cv.transform(novelUsefulSet.values.astype('U')) 



# In[80]:

############ Isolation Forest

print("############# Novelty")

clf = IsolationForest(max_samples="auto")
clf.fit(regularNoveltySet_converted)
predictedResults = clf.predict(requirementTest)
actualResults = np.array(novelty_test)

predictResult = pd.DataFrame(data=actualResults)
#np.savetxt("outputPredicted.txt",predictResult)

result_IsolationForest = classification_report(actualResults,predictedResults)
print("Classification report for Isolation Forest:")
print(result_IsolationForest)

# Confusion matrix novelty
cnf_matrix = confusion_matrix(novelty_test,predictedResults)
tn, fp, fn, tp = confusion_matrix(novelty_test,predictedResults).ravel()
#print("tn:",tn,"fp:",fp, "fn:",fn, "tp:",tp )
print("Confusion Matrix novelty:")
print(cnf_matrix)



print("############# Clarity")
clf = IsolationForest(max_samples="auto")
clf.fit(regularClaritySet_converted)
predictedResults = clf.predict(requirementTest)
actualResults = np.array(clarity_test)


predictResult = pd.DataFrame(data=actualResults)
#np.savetxt("outputPredicted.txt",predictResult)

result_IsolationForest = classification_report(actualResults,predictedResults)
print("Classification report for Isolation Forest:")
print(result_IsolationForest)

# Confusion matrix
cnf_matrix = confusion_matrix(novelty_test,predictedResults)
tn, fp, fn, tp = confusion_matrix(novelty_test,predictedResults).ravel()
#print("tn:",tn,"fp:",fp, "fn:",fn, "tp:",tp )
print("Confusion Matrix novelty:")
print(cnf_matrix)


print("############# Creativity")

clf = IsolationForest(max_samples="auto")
clf.fit(regularCreativeSet_converted)
predictedResults = clf.predict(requirementTest)
actualResults = np.array(creativity_test)


predictResult = pd.DataFrame(data=actualResults)
#np.savetxt("outputPredicted.txt",predictResult)

result_IsolationForest = classification_report(actualResults,predictedResults)
print("Classification report for Isolation Forest:")
print(result_IsolationForest)

# Confusion matrix
cnf_matrix = confusion_matrix(novelty_test,predictedResults)
tn, fp, fn, tp = confusion_matrix(novelty_test,predictedResults).ravel()
#print("tn:",tn,"fp:",fp, "fn:",fn, "tp:",tp )
print("Confusion Matrix novelty:")
print(cnf_matrix)


print("############# Usefulness")
clf = IsolationForest(max_samples="auto")
clf.fit(regularUsefulSet_converted)
predictedResults = clf.predict(requirementTest)
actualResults = np.array(usefulness_test)


predictResult = pd.DataFrame(data=actualResults)
#np.savetxt("outputPredicted.txt",predictResult)

result_IsolationForest = classification_report(actualResults,predictedResults)
print("Classification report for Isolation Forest:")
print(result_IsolationForest)

# Confusion matrix
cnf_matrix = confusion_matrix(novelty_test,predictedResults)
tn, fp, fn, tp = confusion_matrix(usefulness_test,predictedResults).ravel()
#print("tn:",tn,"fp:",fp, "fn:",fn, "tp:",tp )
print("Confusion Matrix novelty:")
print(cnf_matrix)


# print(requirement_test)
# print(type(requirement_test))
# print(predictedResults)



# In[83]:

print("#########################Novelty")

clf = svm.OneClassSVM(nu=0.1,kernel="poly",gamma=0.1)
clf.fit(regularNoveltySet_converted)
svmoneclass = clf.predict(requirementTest)
#print("Predicted values:")
#np.savetxt("outputPredicted.txt",svmoneclass)
#print(svmoneclass)
reqTestDf = pd.DataFrame(data=requirement_test)
#print(reqTestDf)
svmresultDf = pd.DataFrame(data=svmoneclass)
actualResults = np.array(novelty_test)
actualResultsDf = pd.DataFrame(data=actualResults)
print("Actual results:")
#np.savetxt("actualResults.txt",actualResultsDf)
# print(actualResultsDf)
finalResult=pd.DataFrame()
finalResult["requirement"] = reqTestDf
finalResult["PredictionsSVM"] = svmresultDf
finalResult["GroundTruth"] = actualResultsDf

finalResult.to_csv('C:/Users/Anupama/Documents/4thsem/THESIS/Output/Resultsvm.csv',encoding='utf-8')

##result = pd.concat([reqTestDf,svmresultDf,actualResultsDf],ignore_index=True)
#print(result)
print("Classification report for SVM One class")
result = classification_report(actualResults,svmoneclass)
print(result)

cnf_matrix = confusion_matrix(novelty_test,svmoneclass)
tp, fn, fp, tn = confusion_matrix(novelty_test,svmoneclass).ravel()
print("tp:",tp,"fn:",fn, "fp:",fp, "tn:",tn )
print("Confusion Matrix")
print(cnf_matrix)


print("############## Clarity")
clf = svm.OneClassSVM(nu=0.1,kernel="poly",gamma=0.1)
clf.fit(regularClaritySet_converted)
svmoneclass = clf.predict(requirementTest)
actualResults = np.array(clarity_test)
print("Actual results:")

print("Classification report for SVM One class")
result = classification_report(actualResults,svmoneclass)
print(result)

cnf_matrix = confusion_matrix(novelty_test,svmoneclass)
tp, fn, fp, tn = confusion_matrix(clarity_test,svmoneclass).ravel()
print("tp:",tp,"fn:",fn, "fp:",fp, "tn:",tn )
print("Confusion Matrix")
print(cnf_matrix)

print("############## Creativity")
clf = svm.OneClassSVM(nu=0.1,kernel="poly",gamma=0.1)
clf.fit(regularCreativeSet_converted)
svmoneclass = clf.predict(requirementTest)
actualResults = np.array(creativity_test)
print("Actual results:")

print("Classification report for SVM One class")
result = classification_report(actualResults,svmoneclass)
print(result)

cnf_matrix = confusion_matrix(novelty_test,svmoneclass)
tp, fn, fp, tn = confusion_matrix(creativity_test,svmoneclass).ravel()
print("tp:",tp,"fn:",fn, "fp:",fp, "tn:",tn )
print("Confusion Matrix")
print(cnf_matrix)


print("############## Usefulness")

clf = svm.OneClassSVM(nu=0.1,kernel="poly",gamma=0.1)
clf.fit(regularTrainSet_converted)
svmoneclass = clf.predict(requirementTest)
actualResults = np.array(usefulness_test)
print("Actual results:")

print("Classification report for SVM One class")
result = classification_report(actualResults,svmoneclass)
print(result)

cnf_matrix = confusion_matrix(novelty_test,svmoneclass)
tp, fn, fp, tn = confusion_matrix(usefulness_test,svmoneclass).ravel()
print("tp:",tp,"fn:",fn, "fp:",fp, "tn:",tn )
print("Confusion Matrix")
print(cnf_matrix)




# In[ ]:



