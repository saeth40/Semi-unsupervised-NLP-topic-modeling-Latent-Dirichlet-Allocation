import pandas as pd
from pandas import Series, DataFrame
import numpy as np

import joblib
import json

import time
import timeit

# import ENG
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim import corpora, models
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk
nltk.download('wordnet')

import pythainlp
from pythainlp import word_tokenize

print('pythainlp ver:',pythainlp.__version__)



# train
train=pd.read_csv('/home/std/Downloads/csv_file/train_nonStemV2_5000.csv')

# combine descriptoin with job titles
keep=[]
for i in range(len(train)):
    en=train['title_en'].iloc[i]
    th=train['title_th'].iloc[i]
    des=train['description'].iloc[i]
    if type(train['title_en'].iloc[i])!=type('lol'):
        en=''
    if type(train['title_th'].iloc[i])!=type('lol'):
        th=''
    if type(train['description'].iloc[i])!=type('lol'):
        des=''
    keep.append((en+ ' '+th+ ' '+des))
select=pd.Series(keep)
print('len select: ', len(select))


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

#map processAll with the data
start = timeit.default_timer()

processed_docs = select.map(preprocessAll)

joblib.dump(processed_docs, 'processed_docV2')  # save the doc

dictionary = gensim.corpora.Dictionary(processed_docs)

joblib.dump(dictionary, 'dictionary')  # save the dictionary

stop = timeit.default_timer()
print('runime processAll: ', stop - start)
