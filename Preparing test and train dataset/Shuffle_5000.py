import pandas as pd

data=pd.read_csv('C:\\Users\\saeth\\Desktop\\TDRI\\non_stem_all_V2.csv')
#shuffle all
data=data.sample(frac=1)
print('len data', len(data))
#test 5,000 dataset
test=data[:5000].drop(['company'],axis=1)
print('len test', len(test))
test.to_csv('test_nonStemV2_50000.csv',index=False, header=True)

#train
train=data[5000:]
train.to_csv('train_nonStemV2_50000.csv',index=False, header=True)
select=train['description']
print('len select: ', len(select))
