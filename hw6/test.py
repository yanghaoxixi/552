import numpy as np
from cvxopt import matrix, solvers
from sklearn.svm import SVC
import matplotlib.pyplot as plt


a=np.array([1,12,13,14,5,6,7,8])

argsort_a =np.argsort(a)



linsep_data = []
with open("linsep.txt",'r') as file:
    line = file.readline().strip('\n').split(',')
    while line != ['']:
        line = list(map(float, line))
        linsep_data.append(line)
        line = file.readline().strip('\n').split(',')
linnp = np.array(linsep_data)

nonlin_data = []
with open("nonlinsep.txt",'r') as file:
    line = file.readline().strip('\n').split(',')
    while line != ['']:
        line = list(map(float, line))
        nonlin_data.append(line)
        line = file.readline().strip('\n').split(',')
nonnp = np.array(nonlin_data)


x1 = linnp[:,0:2]
y1 = linnp[:,2]
x2 = nonnp[:,0:2]
y2 = nonnp[:,2]


#clf = SVC(kernel='linear',C=1e4)
clf = SVC(kernel='poly',degree=2,C=1e4)
clf.fit(x2,y2)

""" w = clf.coef_.tolist()[0]
b = clf.intercept_.tolist()[0] """

w = clf.dual_coef_.tolist()[0]
b = clf.intercept_.tolist()[0]


lin_po = []
lin_ne = []
for row in nonlin_data:
    if row[2] == 1:               
        lin_po.append(row[0:2])
    else:
        lin_ne.append(row[0:2])
lin_po = np.array(lin_po)
lin_ne = np.array(lin_ne)


plt.scatter(lin_po[:,0],lin_po[:,1], c='r',marker='+')
plt.scatter(lin_ne[:,0],lin_ne[:,1], c='r',marker='_')

plt.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],c='b',marker='.')

""" svp = linnp[sv]
plt.scatter(svp[:,0],svp[:,1],c='b',marker='.') """

""" X = np.linspace(0,1)
Y = -w[0]/w[1] * X +b
plt.plot(X,Y)

plt.xlim(0,1)
plt.ylim(0,1) """

 



plt.show()

a = 1