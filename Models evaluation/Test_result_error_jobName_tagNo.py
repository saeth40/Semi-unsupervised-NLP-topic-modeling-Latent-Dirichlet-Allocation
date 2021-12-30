import os
import pandas as pd
from pandas import DataFrame, Series
import numpy as np

import time
import timeit

import warnings
warnings.filterwarnings('ignore')

raw_path='/home/std/Downloads/python/test_result/tagV1_nonAlphaBeta/'
for raw in sorted(os.listdir(raw_path)):
    start = timeit.default_timer()
    data=pd.read_csv(raw_path+raw)
    allColumn=list(data.columns)
    allModel=list(data.drop(['tag_topic'], axis=1).columns)
    allJob=list(data['tag_topic'].unique())
    frames=[] #concat new csv
    frames_error=[] #concat error csv
    lol=1
    for model in allModel:
        number=pd.DataFrame() #new csv
        error=pd.DataFrame() #collect error
        temp_rec=[] #freq
        dic_number=dict() #for mapping
        dic_job=dict()#new csv
        dup0=[]
        number['model']=pd.Series([model])
        error['model']=pd.Series([model])
        pool=model[24:len(model)-4].split(':')
        error['n_below']=pd.Series([pool[0]])
        error['n_topic']=pd.Series([pool[1]])
        error['alpha']=pd.Series([pool[2]])
        error['eta']=pd.Series([pool[3]])

        #### convert frequency into percentage ####

        for job in allJob:
            dup=[]
            temp=[]
            select=list(data[data['tag_topic']==job][model])
            for i in select:
                if (i not in dup) and (str(i) != 'nan'):
                    temp.append((select.count(i),i,job))
                    dup.append(i)
            gg=sum([i[0] for i in temp])
            tempy=[((a*100)/gg,int(b),c) for (a,b,c) in temp]
            for i in tempy:
                temp_rec.append(i)
        temp2=sorted(temp_rec, reverse=True)

        # dic_job
        for i in range(len(temp2)):
            # new key, new value
            if (temp2[i][2] not in dic_job.keys()) and (temp2[i][1] not in dup0):
                dic_job[temp2[i][2]]=[temp2[i][1]]
                dup0.append(temp2[i][1])
            else:
                #new key, dup_value: skip
                #new value, dup_key: 1. if f1=f2 + value1 = value2/ add new key first, then check len, then sort by charecter
                if (temp2[i][1] not in dup0) and (i != len(temp2)-1) and (temp2[i][0]==temp2[i+1][0]) and (temp2[i][1]==temp2[i+1][1]):
                    if temp2[i+1][2] not in dic_job.keys():
                        dic_job[temp2[i+1][2]]=[temp2[i+1][1]]
                        dup0.append(temp2[i+1][1])
                    elif len(dic_job[temp2[i][2]])>len(dic_job[temp2[i+1][2]]):
                        dic_job[temp2[i+1][2]].append(temp2[i+1][1])
                        dup0.append(temp2[i+1][1])
                    else:
                        dic_job[temp2[i][2]].append(temp2[i][1])
                        dup0.append(temp2[i][1])
                else:
                    # new value, dup key
                    if temp2[i][1] not in dup0:
                        dic_job[temp2[i][2]].append(temp2[i][1])
                        dup0.append(temp2[i][1])
        for i in dic_job.keys():
            number[i]=pd.Series([str(dic_job[i])])
        frames.append(number)

        #dic_number for mapping
        for i in dic_job.keys():
            for j in dic_job[i]:
                if j not in dic_number.keys():
                    dic_number[j]= i

        #mapping
        for i in range(len(data)):
            if str(data[model].iloc[i]) != 'nan':
                if int(data[model].iloc[i]) in dic_number.keys():
                    lol3=int(data[model].iloc[i])
                    data[model].iloc[i]=dic_number[lol3]

        for job in allJob:
            #select=data[data['tag_topic']==job][model].map(dic_number)
            select=list(data[data['tag_topic']==job][model])
            error['error_'+job]=pd.Series((len(select)-select.count(job))/len(select))
        frames_error.append(error)
    # concat the result
    csv2=pd.concat(frames)
    csv3=pd.concat(frames_error)

    # save to csv
    csv2.to_csv('/home/std/Downloads/python/test_result/tagging_percentage_nonAlphaBeta/tagingNumber_'+raw)
    data.to_csv('/home/std/Downloads/python/test_result/tagging_percentage_nonAlphaBeta/tagingJobName_'+raw)
    csv3.to_csv('/home/std/Downloads/python/test_result/tagging_percentage_nonAlphaBeta/taggingError_'+raw)
    stop = timeit.default_timer()
    print('runtime_raw: '+raw+' ', stop - start)
