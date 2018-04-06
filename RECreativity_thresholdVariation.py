
# coding: utf-8

# In[2]:

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


df = pd.read_csv('C:/Users/Anupama/Documents/4thsem/THESIS/requirementsData.csv', sep=',')
df_ratings = pd.read_csv('C:/Users/Anupama/Documents/4thsem/THESIS/requirements_rate.csv', sep=',')
df_cleaned = pd.read_csv('C:/Users/Anupama/Documents/4thsem/THESIS/Output/CleanedRequirementOutput1.csv', sep=',')
df_average = pd.pivot_table(df_ratings,index=["requirement_id"],values=["novelty","usefulness","clarity","creativity_score"],aggfunc=[np.mean])
#df_average.to_csv('C:/Users/Anupama/Documents/4thsem/THESIS/Output/CleanedRequirementOutput1.csv',encoding='utf-8')
df_average["originalRequirement"] = df_cleaned["requirementOriginal"]
df_average["requirement"] = df_cleaned["requirement"]
df_average["domain"]=df["application_domain"]
###### Executed below step and modified the CSV file
#df_average.to_csv('C:/Users/Anupama/Documents/4thsem/THESIS/Output/Requirement_model.csv',encoding='utf-8')

######## Threshold with clarity 4.11 creativity 3.6 novelty 3.3 usefulness 3.83 ####################
df_model = pd.read_csv('C:/Users/Anupama/Documents/4thsem/THESIS/Output/Requirement_model.csv', sep=',')
df_model['clarity'] = np.where(df_model['clarity']>=4.11,-1,1)
df_model['creativity_score'] = np.where(df_model['creativity_score']>=3.6,-1,1)
df_model['novelty'] = np.where(df_model['novelty']>=3.3,-1,1)
df_model['usefulness'] = np.where(df_model['usefulness']>=3.83,-1,1)

df_model.to_csv('C:/Users/Anupama/Documents/4thsem/THESIS/OutputThreshold/Requirement_model1.csv',encoding='utf-8')

######## Threshold with clarity 2.5 creativity 2.5 novelty 2.5 usefulness 2.5 ####################
df_model = pd.read_csv('C:/Users/Anupama/Documents/4thsem/THESIS/Output/Requirement_model.csv', sep=',')
df_model['clarity'] = np.where(df_model['clarity']>=2.5,-1,1)
df_model['creativity_score'] = np.where(df_model['creativity_score']>=2.5,-1,1)
df_model['novelty'] = np.where(df_model['novelty']>=2.5,-1,1)
df_model['usefulness'] = np.where(df_model['usefulness']>=2.5,-1,1)

df_model.to_csv('C:/Users/Anupama/Documents/4thsem/THESIS/OutputThreshold/Requirement_model2_2.5.csv',encoding='utf-8')


######## Threshold with clarity 3.0 creativity 3 novelty 3 usefulness 3 ####################
df_model = pd.read_csv('C:/Users/Anupama/Documents/4thsem/THESIS/Output/Requirement_model.csv', sep=',')
df_model['clarity'] = np.where(df_model['clarity']>=3,-1,1)
df_model['creativity_score'] = np.where(df_model['creativity_score']>=3,-1,1)
df_model['novelty'] = np.where(df_model['novelty']>=3,-1,1)
df_model['usefulness'] = np.where(df_model['usefulness']>=3,-1,1)

df_model.to_csv('C:/Users/Anupama/Documents/4thsem/THESIS/OutputThreshold/Requirement_model3_3.csv',encoding='utf-8')

######## Threshold with clarity 3.5 creativity 3.5 novelty 3.5 usefulness 3.5 ####################
df_model = pd.read_csv('C:/Users/Anupama/Documents/4thsem/THESIS/Output/Requirement_model.csv', sep=',')
df_model['clarity'] = np.where(df_model['clarity']>=3.5,-1,1)
df_model['creativity_score'] = np.where(df_model['creativity_score']>=3.5,-1,1)
df_model['novelty'] = np.where(df_model['novelty']>=3.5,-1,1)
df_model['usefulness'] = np.where(df_model['usefulness']>=3.5,-1,1)

df_model.to_csv('C:/Users/Anupama/Documents/4thsem/THESIS/OutputThreshold/Requirement_model3_3.5.csv',encoding='utf-8')

######## Threshold with clarity 4. creativity 4 novelty 4 usefulness 4 ####################
df_model = pd.read_csv('C:/Users/Anupama/Documents/4thsem/THESIS/Output/Requirement_model.csv', sep=',')
df_model['clarity'] = np.where(df_model['clarity']>=4,-1,1)
df_model['creativity_score'] = np.where(df_model['creativity_score']>=4,-1,1)
df_model['novelty'] = np.where(df_model['novelty']>=4,-1,1)
df_model['usefulness'] = np.where(df_model['usefulness']>=4,-1,1)

df_model.to_csv('C:/Users/Anupama/Documents/4thsem/THESIS/OutputThreshold/Requirement_model4_4.csv',encoding='utf-8')

######## Threshold with clarity 4.5 creativity 4.5 novelty 4.5 usefulness 4.5 ####################
df_model = pd.read_csv('C:/Users/Anupama/Documents/4thsem/THESIS/Output/Requirement_model.csv', sep=',')
df_model['clarity'] = np.where(df_model['clarity']>=4.5,-1,1)
df_model['creativity_score'] = np.where(df_model['creativity_score']>=4.5,-1,1)
df_model['novelty'] = np.where(df_model['novelty']>=4.5,-1,1)
df_model['usefulness'] = np.where(df_model['usefulness']>=4.5,-1,1)

df_model.to_csv('C:/Users/Anupama/Documents/4thsem/THESIS/OutputThreshold/Requirement_model4_4.5.csv',encoding='utf-8')





print(df_model.head(5))

print(df_model.columns)
print(df_average.describe())
print(df_average.columns)




# In[ ]:



