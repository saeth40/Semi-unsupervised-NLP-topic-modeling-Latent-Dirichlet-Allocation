import os
import joblib
import pandas as pd
from pandas import Series, DataFrame
import time
import timeit
import gensim



start = timeit.default_timer()
# load all words table
lol=pd.read_csv('/home/std/Downloads/Allword_FinalDictionaryV1_Final.csv')
select=lol[lol['YorN']=='Y']['word']
unwantedWords=set(select)
print('all Y words: ',len(unwantedWords))

dictionary=joblib.load('/home/std/Downloads/python/dictionary_processed_docV2_includeTitle')
good=0
bad=0
for i in dictionary.values():
    if i in unwantedWords:
        good+=1
    else:
        bad+=1
print('bad= ',bad)
print('good= ',good)

print('len dic before: ',len(dictionary))
allValues=list(dictionary.values())

new_dict=gensim.corpora.Dictionary([list(unwantedWords)])
print('len dic after: ', len(new_dict))

# another method to update a dictionary
# for i in allValues:
#     if i not in unwantedWords:
#         dictionary.filter_tokens(bad_ids=[dictionary.token2id[i]])
# print('len dic after: ',len(dictionary))

joblib.dump(new_dict, '/home/std/Downloads/python/dic_BiTri_eliminateWordV1/dictionary_processed_docV2_includeTitle_eliminatedWordsFinalV2')

stop = timeit.default_timer()
print('runtime_processed_dic: ', stop - start)
