import sys
import csv

train_addr = 'D:/work/AI/python/Predict_PM2.5/train.csv'
test_addr = 'D:/work/AI/python/Predict_PM2.5/test.csv'

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
                train_data[n_row%18-1].append(float(row[idx]))
            else:
                train_data[n_row%18-1].append(0)
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




