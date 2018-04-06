
# coding: utf-8

# In[16]:

import re
import csv
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
from pandas import DataFrame

pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 3000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth',100)
train_req = []
test_req = []
train_ratings = []


def cleantext(text):
    text = str(text).lower()
    text = re.sub('[ ]+[g|G][P|p][s|S][ ]+', ' global positioning system ', text)
    text = re.sub('[ ]+[A|a][/]*[c|C][ ]+', ' air conditioning ', text)
    text = re.sub('[ ]+[t|T][/|\]*[v|V][s|S]*[ ]+', ' television ', text)
    text = re.sub('[0-9][ ]+[p|P][.]+[M|m][.]+', ' night ', text)
    text = re.sub('[0-9][ ]+[a|A][.]+[M|m][.]+', 'morning ', text)
    text = re.sub('[w|W][i|I ][ |/|-|]*[f|F][i|I]', 'wifi', text)
    text = re.sub('[w|W][i|I ][ |/|-|]*[f|F][i|I]', ' wifi ', text)
    text = re.sub('[ ]+[b|B][p|P][ ]+', ' blood pressure ', text)
    text = re.sub('[%]', 'percent', text)
    text = re.sub('[.|//|\|,|;|\'|:|!|(|)|-|$|@|#|]', ' ', text)
    text = text.split()

    return text

def stopwords_text(text):

    stopwordList = set(stopwords.words("english"))
    stopwordList = list(stopwordList)
    stopwordList.append("my")
    stopwordList.append("smart")
    stopwordList.append("house")
    stopwordList.append("home")

    for word in list(text):
        if word in stopwordList:
            text.remove(word)
    return text

def stem_text(text):
    stemmedSentence =[]
    ps = PorterStemmer()
    for words in list(text):
        text = ps.stem(words)
        stemmedSentence.append(text)
    return stemmedSentence

def main():
    # result_df = pd.DataFrame()
    df = pd.read_csv('C:/Users/Anupama/Documents/4thsem/THESIS/requirementsData.csv', sep=',')
    df_ratings = pd.read_csv('C:/Users/Anupama/Documents/4thsem/THESIS/requirementsData.csv', sep=',')
    
    sentences=[]
    

    for text in df['feature']:
        text_cleaned=cleantext(text)
        text_stopword = stopwords_text(text_cleaned)
        stemmedWord = stem_text(text_stopword)
        sentences.append(stemmedWord)

    df_cleaned = pd.DataFrame({'requirement': sentences})
    df_cleaned['requirementOriginal']=df['feature']
    df_cleaned['requirement'] = df_cleaned.requirement.apply(' '.join)
    df_cleaned.to_csv('C:/Users/Anupama/Documents/4thsem/THESIS/Output/CleanedRequirementOutput1.csv',encoding='utf-8')

if __name__ == '__main__':
    main()
    
    


# In[ ]:




# In[ ]:



