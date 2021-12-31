import pandas as pd
from pandas import Series, DataFrame
import numpy as np

import joblib
import json

import time
import timeit

#import ENG
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk
nltk.download('wordnet')

import os

import pythainlp
from pythainlp import word_tokenize

print('pythainlp ver:',pythainlp.__version__)



#print(select)
from gensim import corpora, models


# all function

#function to solve: ManagerFor, managerFor
#input = a word (only English)
#output = [w1, w2, w3,...]
import re
def splitword(text):
  out=[]  # output: list of string
  couUp=0  # count of upper cases
  left=0  # left index
  initial=text.strip()
  upper=re.findall('[A-Z]',initial)  # find all upper cases
  for i in range(len(initial)):
    if initial[i] in upper:  # if char is upper case
      couUp+=1
    if couUp==len(upper):
      out.append(initial[left:i])
      out.append(initial[i:])
      break
    elif initial[i] in upper and i !=0:  # if char is upper case and not the first char
      out.append(initial[left:i])
      left=i
  return ' '.join(out).strip()

#Data Pre-processing for English only
# input = many words
#output = [w1, w2, w3, ......]

#1. Tokenization: Split the text into sentences and the sentences into words.
# Lowercase the words and remove punctuation.
#2. Words that have fewer than 3 characters are removed.
#3. All stopwords are removed.
#4. Words are lemmatized — 
#words in third person are changed to first person and verbs in past and future tenses are changed into present.
#5. Words are stemmed — words are reduced to their root form.
def lemmatize_stemming(text):
    return SnowballStemmer('english').stem(WordNetLemmatizer().lemmatize(text, pos='v'))
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

#process for all data
#input = docs
# output = a list of each word
def preprocessAll(text):
  text=str(text)
  text=text.strip()
  thai=pythainlp.thai_characters  # all Thai chars
  th=''
  en=''
  for i in range(len(text)):
    if text[i] in thai:  # if char in Thai
      th=th+text[i]
    else:  # if char in Eng
      if (i != 0) and (text[i-1] in thai):  # if char is not the first char, and the previous char is Thai
        en=en+' '+text[i]
      else:  
        en=en+text[i]
  temp_en=[]  # record Eng chars
  for i in en.split(' '):
    if i !='':
      temp_en.append(splitword(i))
  temp_en=' '.join(temp_en)
  return preprocess(temp_en)+word_tokenize(th, keep_whitespace=False)

# prepare test data
test=pd.read_csv('/home/std/Downloads/python/test_dataset_desPlusTitleV1.csv')[['job_label','combine']]
test.dropna(inplace=True)
test.reset_index(drop=True, inplace=True)
print(len(test))
print(test.index)

# bigram, trigram
bigram=joblib.load('/home/std/Downloads/python/bigram_processed_docs_V2_includeTitle')
trigram=joblib.load('/home/std/Downloads/python/trigram_processed_docs_V2_includeTitle')
start = timeit.default_timer()
dic_word=dict()
for i in range(len(test['combine'])):
    temp=[]
    lol=preprocessAll(test['combine'].iloc[i])
    bi=bigram[lol]
    tri=trigram[bigram[lol]]
    for w1 in bi:
        if w1 not in lol:
            temp.append(w1)
    for w2 in tri:
        if w2 not in lol:
            temp.append(w2)
    for word in temp:
 
        if word not in dic_word.keys():
            dic_word[word]=[[test['job_label'].iloc[i], 1]]
        else:
           allJob=[tup[0] for tup in dic_word[word]]
            if test['job_label'].iloc[i] not in allJob:
                dic_word[word].append([test['job_label'].iloc[i], 1])
            else:
                for g in range(len(dic_word[word])):
                    if dic_word[word][g][0] == test['job_label'].iloc[i]:
                        dic_word[word][g][1]+=1
                         
joblib.dump(dic_word, 'dic_allTestWord_BiTri_TitleV1')
 
stop = timeit.default_timer()
print('dic finished at: ', stop-start)

# record all words and create 'YorN' column
start = timeit.default_timer()
frame=[]
for word in dic_word.keys():
    out=pd.DataFrame()
    out['word']=pd.Series([word])
    for job in test['job_label'].unique():
        job_list=[tup[0] for tup in dic_word[word]]
        if job not in job_list:
            out[job]=pd.Series([0])
        else:
            for item in dic_word[word]:
                if job == item[0]:
                    out[item[0]]=pd.Series([item[1]])
    frame.append(out)
allData = pd.concat(frame)
allData['YorN']=pd.Series()
allData.to_csv('/home/std/Downloads/python/AllWord_FromTestData_BiTri_Title_V1'+'.csv',index=False, header=True)
stop = timeit.default_timer()
print('csv finished at: ', stop-start)

# calculate percentage of each word
select=allData.drop(['word'],axis=1)
for job in select.columns:
    allData[job]=(allData[job]*100)/sum(allData[job])
allData.to_csv('/home/std/Downloads/python/AllWord_Percent_FromTestData_BiTri_Title_V1'+'.csv',index=False, header=True)



'''#threshole'''
target='accountant + audit'
data=test=pd.read_csv('/home/std/Downloads/python/AllWord_Percent_FromTestData_noBiTri_Title_V1.csv')
for target in data.drop(['word'],axis=1).columns:
    data=test=pd.read_csv('/home/std/Downloads/python/AllWord_Percent_FromTestData_noBiTri_Title_V1.csv')
    test=pd.read_csv('/home/std/Downloads/python/AllWord_Percent_FromTestData_noBiTri_Title_V1.csv')[target]
    test.reset_index(drop=True, inplace=True)
    print(len(test))
    print(test.index)
    print(data.index)
 
    thres=0.1
    print('thres: ', thres)
    rec=[]
    for i in range(len(test)):
        if test.iloc[i]<= thres:
           rec.append(i)
    for h in rec:
        data.drop(index=h, inplace=True)
    data=data[['word',target]]
    data.to_csv('/home/std/Downloads/python/All_job_keywords/AllWord_' +target+ '_Thres_' +str(thres)+ '_Percent_FromTestData_noBiTri_Title_V1'+'.csv',index=False, header=True)
    print('len after: ', len(data))


