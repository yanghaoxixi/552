import random
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn import datasets
from sklearn.linear_model.logistic import LogisticRegression
from sklearn import linear_model
import matplotlib.pyplot as plt

iris=datasets.load_iris()
X=iris.data[:,[2,3]]
y=iris.target


data_clas = []                                              # obtain data for classification
with open('classification.txt','r') as file:
    line = file.readline().strip('\n').split(',')
    while line != ['']:
        line = list(map(float, line))
        data_clas.append(line)
        line = file.readline().strip('\n').split(',')
dataclas_np = np.array(data_clas)  
mean = dataclas_np[:,0:3].mean(axis=0)
sigma = dataclas_np[:,0:3].std(axis=0)
dataclas_np[:,0:3] = (dataclas_np[:,0:3]-mean)/sigma

XX = dataclas_np[:,0:3]
yy = dataclas_np[:,4]

clf=Perceptron(max_iter=7000,eta0=0.1)
clf.fit(dataclas_np[:,0:3],dataclas_np[:,4])
#clf.fit(X,y)
weights=clf.coef_

bias=clf.intercept_

""" xm = np.mat(XX).T
ym = np.mat(weights) * xm + bias
ym = np.sign(ym) """
acc=clf.score(XX,yy)*100

y_pre = clf.predict(XX)
d = np.equal(y_pre, yy)
dd = np.sum(np.equal(y_pre, yy)==True)
print('matchs:{0}/{1}'.format(np.sum(np.equal(y_pre, yy)==True), yy.shape[0]))
print(dd/yy.shape[0])


t = np.array([1,2,3])
u = np.array([4,5,6])
print(t*u.T)

claa = LogisticRegression(max_iter=7000)
claa.fit(XX,yy)
acc2 = claa.score(XX,yy)

y_pre = claa.predict(XX)
d = np.equal(y_pre, yy)
dd = np.sum(np.equal(y_pre, yy)==True)
print('matchs:{0}/{1}'.format(np.sum(np.equal(y_pre, yy)==True), yy.shape[0]))
print(dd/yy.shape[0])



data_reg = []                                               # obtain data for linear regression
with open('linear-regression.txt','r') as file:
    line = file.readline().strip('\n').split(',')
    while line != ['']:
        line = list(map(float, line))
        data_reg.append(line)
        line = file.readline().strip('\n').split(',')
datareg_np = np.array(data_reg)

xl = datareg_np[:,0:2]
yl = datareg_np[:,2]

lr = linear_model.LinearRegression()
lr.fit(xl,yl)
w = []
w.append(lr.intercept_)
w.append(lr.coef_.tolist()[0])
w.append(lr.coef_.tolist()[1])

x = np.arange(10)
 
fig = plt.figure()
ax = plt.subplot(111)
 
plt.plot(np.arange(1,1200,1),np.arange(1,1200,1),label='PLA',color='r')
 

 
plt.show()

a = 1