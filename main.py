# @File    : main.py
# @Date    : 2021-8-6
# @Author  : Langqing Zou
# @Software: VS Code
# @Python Version: python 3.6

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

'''--------------------Read data and Preprocessing--------------------'''  
train_data = pd.read_csv("./train.csv")
temp = pd.read_csv("./train.csv")
# delete "date" and "station" column
# First way to delete a column
del train_data["Date"]
del train_data["stations"]
# Second way to delete a column
# train_data.drop(['Date', 'stations'], axis=1, inplace=True)
column = train_data["observation"].unique()

# Try to have more train data:
# Train data: 0-9,1-10,2-11.....; check it with 10,11,12 respectively
features = 18
dataTrain = []
dataCheck = []
temp.drop(['Date', 'stations', 'observation'], axis=1, inplace=True)
for i in range(240): # 240 days
    day = temp[i*features:(i+1)*features]
    # 0-9,1-10,2-11....., 15 sets totally
    for j in range(15):
        train = day.iloc[:,j:j + 9] # iloc[row,column]
        check = day.iloc[9,j+9]
        dataTrain.append(train)
        dataCheck.append(check)

# print(dataTrain[1])

'''--------------------Observe the relation--------------------'''

'''Step1: create a dataframe'''
new_frame = pd.DataFrame(np.zeros([24*240, 18]), columns=column) # 240 days
for i in column:
    data = train_data[train_data["observation"]==i]
    data[data == 'NR'] = '0'
    del data["observation"]
    data = np.array(data)
    data = data.astype('float') # str -> float
    data = data.reshape(1,24*240)
    data = data.T
    new_frame[i] = data
# print(new_frame)

'''Step2: using heatmap'''
# fig = plt.figure(figsize=(9, 6))
# ax = fig.add_subplot()
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(new_frame.corr(), fmt="d", linewidths=0.5, ax=ax)
# plt.show()

# From the heatmap, we found that PM2.5,PM10 and NO2 has the most strong relation of prediction with PM2.5
# So we choose these three features to create the model


'''--------------------Create model--------------------'''
# Train the model with first 9 hours every day, then check it with the PM2.5 in the tenth hour
# we assume the model will be relative with the three features mentioned above
# Y = b + W1X1 + W2X2 + ...... + W27X27
interation_count = 10000
learning_rate = 0.00001
b = 0.0001
parameters=[0.001]*27 # all the Ws
loss_history=[]
dict={0:8,1:8,2:8,3:8,4:8,5:8,6:8,7:8,8:8,9:9,10:9,11:9,12:9,13:9,14:9,15:9,16:9,17:9,18:12,19:12,20:12,21:12,22:12,23:12,24:12,25:12,26:12}
iteration_count = 10000  #迭代次数
for i in range(interation_count):
    # initial bias gradient and weight gradient
    b_grad=0
    w_grad=[0]*27
    loss = 0
    num = range(0, 3600)   # 240 days * 15 sets
    nums = list(random.sample(num, 100)) # choose any 100 samples
    # print(nums)
    sample = 100
    for k in range(sample):
        index = nums.pop()
        oneDay = dataTrain[index]
        partsum = b+parameters[0]*float(day.iloc[8,0])+ parameters[1]*float(day.iloc[8,1])+\
                  parameters[2]*float(day.iloc[8,2])+ parameters[3]*float(day.iloc[8,3])+ \
                  parameters[4]*float(day.iloc[8,4])+ parameters[5]*float(day.iloc[8,5])+ \
                  parameters[6]*float(day.iloc[8,6])+ parameters[7]*float(day.iloc[8,7])+ \
                  parameters[8]*float(day.iloc[8,8])+ parameters[9]*float(day.iloc[9,0])+ \
                  parameters[10]*float(day.iloc[9,1])+ parameters[11]*float(day.iloc[9,2])+ \
                  parameters[12]*float(day.iloc[9,3])+ parameters[13]*float(day.iloc[9,4])+ \
                  parameters[14]*float(day.iloc[9,5])+ parameters[15]*float(day.iloc[9,6])+ \
                  parameters[16]*float(day.iloc[9,7])+ parameters[17]*float(day.iloc[9,8])+ \
                   parameters[18]*float(day.iloc[12,0])+ parameters[19]*float(day.iloc[12,1])+ \
                   parameters[20]*float(day.iloc[12,2])+ parameters[21]*float(day.iloc[12,3])+ \
                   parameters[22]*float(day.iloc[12,4])+ parameters[23]*float(day.iloc[12,5])+ \
                   parameters[24]*float(day.iloc[12,6])+ parameters[25]*float(day.iloc[12,7])+ \
                   parameters[26]*float(day.iloc[12,8])- float(dataCheck[index])
        loss = loss + partsum * partsum
        b_grad = b_grad + partsum
        for k in range(27):
            w_grad[k]=w_grad[k]+ partsum * float(day.iloc[dict[k],k % 9])
    loss_history.append(loss/2)
    # update b and w
    b = b - learning_rate * b_grad/sample
    for t in range(27):
        parameters[t] = parameters[t] - learning_rate * (w_grad[t]/sample)

print(b)
print("-----------------------------------")
print(parameters)


'''--------------------Test model--------------------'''