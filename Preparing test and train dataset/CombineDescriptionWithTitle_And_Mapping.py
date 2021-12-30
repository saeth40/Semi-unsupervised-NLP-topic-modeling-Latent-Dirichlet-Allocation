import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import os

import warnings
warnings.filterwarnings('ignore')


#find list of csv file

path='C:\\Users\\saeth\\Desktop\\TDRI\\test_csv'
allCsv=os.listdir(path)
frames = []  # to collect output dataframe

#mapping dictionary from numbers to job titles
dic_pool={'1.1':'architecture', '1.2':'interior archetecture','2.1':'accountant + audit','2.2':'investor + trader + risk management',
          '2.3.1':'marketing', '2.3.2':'pr + ae', '2.3.3':'sale', '2.3.4':'telesale', '2.4.1':'stock + inventory', '2.4.2':'procurement + logistic',
          '3':'law','4':'HR', '5.1.1':'teacher academics', '5.1.2':'teacher sport', '5.1.3':'teacher music', '5.1.4':'teacher art', '5.2':'customer service',
          '6.1.1':'drama writer', '6.1.2':'writing creator', '6.2':'graphic design', '6.3.1':'flim', '6.3.2':'photographer', '6.4':'craftman',
          '7.1':'coordinator high skill', '7.2':'coordinator low skill', '8.1':'waiter', '8.2':'maid', '9':'translator', '10':'vehicle driver',
          '11':'cooker + bartender', '12':'librarian', '13.1':'entertainer', '13.2':'announcer + mc', '13.3':'tour guider', '14':'gardener',
          '15.1':'security + safety', '15.2':'care taker', '15.3':'therapist',
          '16.1':'product', '16.2.1':'retail manager', '16.2.2':'finance manager', '16.2.3':'hr manager', '16.2.4':'policy and planing manager',
          '16.2.5':'business service and admin manager', '16.2.6':'sale and marketing manager', '16.2.7':'advertising and public relation manager',
          '16.2.8':'r and d manager', '16.2.9':'manufacturing manager', '16.2.10':'supply and distribution manager', '16.2.11':'childcare service manager',
          '16.2.12':'health service manager', '16.2.13':'elderly service manager','16.2.14':'social welfare manager', '16.2.15':'education manager',
          '16.2.16':'other managers', '16.3':'event organizer', '17':'buesiness develop', '18':'credit analyst', '19':'cashier','20':'artist'}

for i in allCsv:
    data=pd.read_csv(path+'\\'+i, skiprows=0)
    data2=data[['_id','description','title_en','title_th', 'job_label']]  # choose column job_description and job_label
    print('file name: ',i)
    print('len before', len(data2))
    

    
    #print('len before drop na', len(data2))
    
    #drop na
    data2.dropna(subset=['job_label'], inplace=True)
    data2.reset_index(drop=True, inplace=True)
    print('len after drop na', len(data2))
    keep=[]  # record string in 'combine' column
    for i in range(len(data2)):
        en=data2['title_en'].iloc[i]  # title_en
        th=data2['title_th'].iloc[i]  # title_th
        des=data2['description'].iloc[i]  # description

        # if selected record is not string, define it as ''
        if type(data2['title_en'].iloc[i])!=type('lol'):
            en=''
        if type(data2['title_th'].iloc[i])!=type('lol'):
            th=''
        if type(data2['description'].iloc[i])!=type('lol'):
            des=''
        keep.append((en+' '+th+' '+des))  # record the result
    data2['combine']=pd.Series(keep, index= data2['job_label'].index)  # update the keep to 'combine' column
    print(data2['combine'][0:4])
    frames.append(data2)  # record dataframe into list frames
    
allData = pd.concat(frames)  # concat all dataframe
print('len total: ',len(allData))

allData['job_label'] = allData['job_label'].astype(str)  # define type of job_label to string
print(list(allData['job_label'].unique()))

# deal with human errors
allData['job_label']=allData['job_label'].replace(['-','?','Nonsense','Nonsence','consult'], np.nan)
allData['job_label']=allData['job_label'].replace(['STEM'], ['stem'])
allData['job_label']=allData['job_label'].replace(['5.1'], ['5.1.1'])
allData['job_label']=allData['job_label'].replace(['16'], ['16.1'])
allData['job_label']=allData['job_label'].replace(['6.2.10'], ['16.2.10'])
allData['job_label']=allData['job_label'].replace(['2.33'], ['2.3.3'])

# drop na and reset index
allData.dropna(subset=['job_label'],inplace=True)
allData.reset_index(drop=True, inplace=True)

print('len total after: ',len(allData))
print(sorted(list(allData['job_label'].unique())))

#maping numbers to job titles
key=dic_pool.keys()
for i in range(len(allData)):
    if allData['job_label'].iloc[i] in key:
        allData['job_label'].iloc[i] = dic_pool[allData['job_label'].iloc[i]]
print('no_jobs: ',len(list(allData['job_label'].unique())))
print('no_data: ',len(allData))
print(allData['job_label'].value_counts())
print(sorted(list(allData['job_label'].unique())))


allData.to_csv('test_dataset_desPlusTitleV1'+'.csv',index=False, header=True)  # save to csv

