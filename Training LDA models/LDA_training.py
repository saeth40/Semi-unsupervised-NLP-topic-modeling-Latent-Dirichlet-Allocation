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

import pythainlp
from pythainlp import word_tokenize

print('pythainlp ver:',pythainlp.__version__)


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


# load processed doc
start = timeit.default_timer()
processed_docs=joblib.load('/home/std/Downloads/python/processed_docs_BiTriV1/processed_docs_BiTri_min5thres10V1')
stop = timeit.default_timer()
print('runtime_processed_docs: ', stop - start)


#fillter words that appear less than no_bow docs + more than no_above (fraction)
# and keep the first most frequent words = keep_n
for keep in [50000, 75000, 100000, 150000, 175000, 200000]:
    for below in [5,10, 15, 20, 25, 30, 35, 40 ,45, 50]:
        for above in [0.1, 0.3, 0.5, 0.7, 0.9]:
            dictionary=joblib.load('/home/std/Downloads/python/dic_BiTri_eliminateWordV1/dic_eliminatedWord_min5thres10V1')
            dictionary.filter_extremes(no_below=below, no_above=above, keep_n=keep)
            print('len dic filler: '+str(len(dictionary))+' below: '+str(below))
            #doc2bow = list of (id of a word, no_appear in a doc)
            
            bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

            """# modelling"""

            #model for calculate weight(related to fraction of docs that a word occur vs a word frequency in a doc)
            #extract main words that represent the doc 
            # weight = (a word frequency in a doc) + inv(doc frequency)
            # ex assume topic = [salad, pizza], f(food) high in doc frequency (food occur in all topics)
            # but vegetable occur most only in salad so f(vegetable) only high in a word frequency but low in doc frequency 
            
            tfidf = models.TfidfModel(bow_corpus)
            corpus_tfidf = tfidf[bow_corpus]

 
            #run model using corpus_tfidf

            for topic in [i for i in range(1, 121)]:
                for alp in [i/1000 for i in range(1, 101, 1)].append('symmetric'):
                    for beta in [i/100 for i in range(1, 1001, 1)].append('None'):
                        start = timeit.default_timer()
                        lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, id2word=dictionary, num_topics=topic, alpha=alp, eta=beta)


                        """# save the model"""

                        model=lda_model_tfidf
                        filename = '/home/std/Downloads/python/model_eliminatedWords_V1/eliminatedWord_min5thres10[keep:below:above:topic:alp:beta]_' +\
                                   str(keep)+':'+str(below)+':'+str(above)+':'+str(topic)+':'+str(alp)+':'+str(beta) + '.sav'
                        joblib.dump(model, filename)
                        stop = timeit.default_timer()
                        print('runime model_tfidf[below:topic]_'+str(below)+':'+str(topic)+': ' , stop - start)
                
