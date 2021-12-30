import pandas as pd

data=pd.read_csv('C:\\Users\\saeth\\Desktop\\TDRI\\test_nonStemV2_5000.csv')
print(data.columns)
div=500  # division
num_people=10
left=500  # slicing index
for i in range(2,num_people+1):  # i = 1,2,3,..., num_people
    select=data[left:i*div]  # selected rows
    select['job_label']=pd.Series()  # add 'job_label' column
    print('i=',i)
    print(left,i*div)
    select.to_csv('C:\\Users\\saeth\\Desktop\\TDRI\\csv_5000\\test_5000_no'+str(i-1)+'.csv',index=False, header=True)
    left=i*div  # update slicing index
