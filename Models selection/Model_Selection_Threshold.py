import pandas as pd
from pandas import DataFrame, Series
import numpy as np

import warnings
warnings.filterwarnings('ignore')

import time
import timeit

start = timeit.default_timer()
# define threshold
thres20=['sale // service', 'marketing // advertising', 'manufacturing manager', 'waiter', 'retail manager', 'Policy manager // BD', 'graphic design', 'writer', 'interior archetecture',
        'architecture']
thres50=['sale // service', 'Policy manager // BD']
thres70=['marketing // advertising']

frames=[]
name=pd.read_csv('/home/std/Downloads/python/test_result/tagingJobName_Title_noBiTri_Final%V3_AlphaBeta_tag2_V8.csv')
rep=pd.read_csv('/home/std/Downloads/python/test_result/tagingNumber_Title_noBiTri_Final%V3_AlphaBeta_tag2_V8.csv')
for model in sorted(list(name.drop(['Unnamed: 0','tag_topic'], axis=1).columns)):
    rec=dict()
    #for job_main in sorted(list(name['tag_topic'].unique())):
    #for job_main in sorted(thres20):
    for job_main in thres50:
        for job in sorted(list(name['tag_topic'].unique())):
            temp=list(name[name['tag_topic']==job][model])
            err=(temp.count(job_main)/len(temp))*100
            if job_main == job:
                if err<70:
                    if job_main in rec.keys():
                        del rec[job_main]
                    break
                else:
                    if job_main not in rec.keys():
                        rec[job_main]=[[job, err]]
                    else:
                        rec[job_main].append([job, err])
            else:
                if err>50:
                    if job_main in rec.keys():
                        del rec[job_main]
                    break
                else:
                    if job_main not in rec.keys():
                        rec[job_main]= [[job, err]]
                    else:
                        rec[job_main].append([job, err])
    if len(rec) !=0:
        for key in rec.keys():
            out=pd.DataFrame()
            out['job']=pd.Series([key])
            for item in rec[key]:
                out['in_'+item[0]]=pd.Series([item[1]])
                #print(out['in_'+item[0]])
            lol=[f[1] for f in rec[key] if f[0] != key]
            out['avg_err']= pd.Series([sum(lol)/len(lol)])
            text5=model[24:len(model)-4].split(':')
            out['alpha']=pd.Series([text5[2]])
            out['beta']=pd.Series([text5[3]])
            out['n_topic']=pd.Series([text5[1]])
            out['represent_topic']=pd.Series([rep[rep['model']==model][job_main].iloc[0]])
            #print(out['represent_topic'])
            frames.append(out)
allData=pd.concat(frames)
allData.to_csv('/home/std/Downloads/python/test_result/model_selection_thres70_50_tag2_V8.csv')
        
stop = timeit.default_timer()
print('runtime: ', stop - start)

                        
