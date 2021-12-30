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
from gensim.models.phrases import Phrases
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


# Bigram, Trigram
processed_docs=joblib.load('processed_docV1')  # load processed doc
start = timeit.default_timer()
minCou=5  # minimum counting of Bigram, Trigram
thres=20  # threshold of Bigram, Trigram
bigram = gensim.models.phrases.Phrases(processed_docs, min_count=minCou, threshold=thres)
trigram = gensim.models.phrases.Phrases(bigram[processed_docs], threshold=thres)
# Add Bigram to doc
for i in range(len(processed_docs)):
    for j in bigram[processed_docs[i]]:
        if j not in processed_docs[i]:
            processed_docs[i].append(j)
# Add Trigram to doc
for i in range(len(processed_docs)):
    for j in trigram[bigram[processed_docs[i]]]:
        if j not in processed_docs[i]:
            processed_docs[i].append(j)
print(processed_docs[0])

# save data
joblib.dump(processed_docs, 'processed_docs_BiTri_min' + str(minCou)+'thres'+ str(thres)+'V1')  # save doc
dictionary = gensim.corpora.Dictionary(processed_docs)
joblib.dump(dictionary, 'dictionary_BiTri_min' + str(minCou)+'thres'+ str(thres)+'V1')  # save dictionary
stop = timeit.default_timer()
print('runtime_processed_docs: ', stop - start)
