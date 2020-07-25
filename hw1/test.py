import math
import copy
import random
import numpy as np
from sklearn import tree

attri = []                                       # Attributes
dataMatrix = []                                  # raw data
with open('dt_data.txt','r') as data:
    line = data.readline()
    line = line.strip('()\n').replace(" ","").split(',')
    attri = line
    line = data.readline()                         #first line -- attribute
    while line:                                     # get data
        if line == '\n':                            # ignore 2nd line which is empty
            line = data.readline()
            continue
        line = line[4:].strip(';\n').replace(" ","").split(',')
        dataMatrix.append(line)
        line = data.readline()

#transfered txt into list data above

row = len(dataMatrix)
column = len(attri) - 1
attri_size = {}                                         #record the number of each terms
dataMatrix_index = copy.deepcopy(dataMatrix)
dataMatrix_trans = {}
for i in range(column):
    memory = []
    dataMatrix_trans[attri[i]] = []
    for j in range(row):
        if dataMatrix[j][i] not in memory:
            memory.append(dataMatrix[j][i])
            dataMatrix_trans[attri[i]].append(dataMatrix[j][i])
            dataMatrix_index[j][i] = memory.index(dataMatrix[j][i])
        else:
            dataMatrix_index[j][i] = memory.index(dataMatrix[j][i])
    attri_size[attri[i]] = len(memory)
dataOper = copy.deepcopy(dataMatrix_index)
attriOper = copy.deepcopy(attri)

X = []
Y = []
for rows in dataOper:
    X.append(rows[:-1])
    if rows[-1] == 'No':
        Y.append(0)
    else:
        Y.append(1)
xx = np.array(X)
yy = np.array(Y)

clf = tree.DecisionTreeClassifier(criterion='entropy')
print(clf)
clf.fit(xx,yy)
print(clf.predict([[1,2,0,1,0,0]]))

a = 1