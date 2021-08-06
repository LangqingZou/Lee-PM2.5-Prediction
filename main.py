# @File    : main.py
# @Date    : 2021-8-6
# @Author  : Langqing Zou
# @Software: VS Code
# @Python Version: python 3.6

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

'''--------------------Read data and Preprocessing--------------------'''  
train_data = pd.read_csv("./train.csv")
# delete "date" and "station" column
# First way to delete a column
del train_data["Date"]
del train_data["stations"]
# Second way to delete a column
# train_data.drop(['Date', 'stations'], axis=1, inplace=True)
column = train_data["observation"].unique()


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
plt.show()














'''--------------------Create model--------------------'''
'''--------------------Test model--------------------'''