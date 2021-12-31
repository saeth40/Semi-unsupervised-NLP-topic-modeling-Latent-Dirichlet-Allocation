import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import os

import time
import timeit

start = timeit.default_timer()
path='/home/std/Downloads/python/test_result/model_selection_list/'
frame1=[]
for file in sorted(os.listdir(path)):
    data=pd.read_csv(path+file)
    #print(len(data))
    frame1.append(data)
allData=pd.concat(frame1)
allData.reset_index(drop=True, inplace=True)
for job in sorted(list(allData['job'].unique())):
    temp=[]
    select=allData[allData['job']==job]
    acc=list(select['in_'+job])
    err=list(select['avg_err'])
    #print(acc)
    #print(err)
    avg_acc=sum(acc)/len(acc)
    avg_err=sum(err)/len(err)
    #print(avg_acc)
    #print(avg_err)
    for i in range(len(select)):
        #print(select.iloc[i]['in_'+job])
        #print(select.iloc[i]['avg_err'])
        if select.iloc[i]['in_'+job]>=avg_acc:
            score_acc= ((select.iloc[i]['in_'+job]-avg_acc)/(max(acc)-avg_acc))*1.5
        else:
            score_acc=((select.iloc[i]['in_'+job]-avg_acc)/(avg_acc- min(acc)))*1.5
        if select.iloc[i]['avg_err']>=avg_err:
            score_err=((select.iloc[i]['avg_err']-avg_err)/(max(err)-avg_err))*(-1)
        else:
            score_err=((select.iloc[i]['avg_err']-avg_err)/(avg_err- min(err)))*(-1)
        temp.append((score_acc+score_err, list(select.index)[i]))
    lol=sorted(temp, reverse= True)
    g=[t[1] for t in lol]
    for i in g:
        if i != g[0]:
            allData.drop(index=i, inplace=True)
        
allData.reset_index(drop=True, inplace=True)
allData.to_csv('/home/std/Downloads/python/test_result/model_selection_From_thres_tag2_V8.csv')

stop = timeit.default_timer()
print('runtime: ', stop - start)
