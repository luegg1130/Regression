import sys
import csv
import numpy as np

train_addr = 'D:/python/code/Regression/train.csv'
test_addr = 'D:/python/code/Regression/test.csv'

#read data
train_data = []
for i in range(18):
    train_data.append([])

txt = open(train_addr, 'r', encoding='big5', newline='')
rows = csv.reader(txt , delimiter=",")

n_row = 0
for row in rows:
    if(n_row != 0):
        for idx in range(3, 27):
            if(row[idx] != 'NR'):
                train_data[(n_row-1)%18].append(float(row[idx]))
            else:
                train_data[(n_row-1)%18].append(0)
    n_row = n_row + 1

txt.close()

#parse data to (x,y)
x = []
y = []
for mon in range(12):
    for data_m in range(471):
        x.append([])
        for data_type in range(18):
            for hr in range(9):
                x[mon*471+data_m].append(train_data[data_type][mon*480+data_m+hr])
        y.append(train_data[9][mon*480+data_m+9])
x = np.array(x)
y = np.array(y)

#add bias
x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
print(x[0])

#init weight & hyperparams
w = np.zeros(len(x[0]))
l_rate = 10
repeat = 10000




