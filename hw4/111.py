import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import copy


def sigmoid(s):         
    return np.exp(s)/(1 + np.exp(s))
def predit(x,weight):
    return sigmoid(np.dot(weight,x))
def cost(x,y,weight):
    oywx = np.multiply(y,np.array(np.dot(weight,x)))
    ss = np.log(1/sigmoid(oywx))
    return np.sum(ss)/len(x.T)
def gradient(x,y,w):
    oywx = np.multiply(y,np.array(np.dot(w,x)))
    h = np.multiply(y,(sigmoid(oywx) - 1))
    grad = (np.multiply(h,x).sum(axis=1))/len(x.T)
    
    return grad
def PLA(data,x_len,y_col,algorithm="PLA"):                                  # implement PLA, x_len means first x_len column is for x, y_col is the column for y
    size = len(data)
    data = np.array(data)
    x0 = np.mat([1 for i in range(size)]).T
    x = np.hstack((x0,data[:,0:x_len])).T
    y = data[:,y_col-1]
    w = np.array([random.uniform(-1,1) for i in range(x_len + 1)])       #got x, y and initialize w with size N+1, +1 for w0
    r = np.sign(np.array(w*x))
    deter_arg = np.argwhere((r[0]==y) == False)
    deter_arglist = [deter_arg[i][0] for i in range(len(deter_arg))]

    
    if algorithm == "PLA":
        footStep = 0.1
        loop = 0
        while deter_arglist != []:
            pick = random.randint(0,len(deter_arglist)-1)
            w = w + footStep * y[deter_arglist[pick]] * x[:,deter_arglist[pick]].T
            
            
            r = np.sign(np.array(w*x))
            deter_arg = np.argwhere((r[0]==y) == False)
            deter_arglist = [deter_arg[i][0] for i in range(len(deter_arg))]
            acc = (1 - len(deter_arglist)/size) * 100 
            loop += 1
        return w.tolist()[0], acc
    elif algorithm == "Pocket":
        loop = 0
        w_best = copy.deepcopy(w)
        mistake = len(deter_arglist)
        footStep = 0.1
        record = []
        record_best = []
        while loop < 7000:
            if len(deter_arglist) == 0:
                return w_best.tolist()[0], 1- (mistake/size),record,record_best
            loss = y * np.array(w*x)
            pp = np.argmin(loss)
            mi = loss[0,pp]
            pick = random.randint(0,len(deter_arglist)-1)
            w = w + footStep * y[deter_arglist[pick]] * x[:,deter_arglist[pick]].T
            #w = w + footStep * y[pp] * x[:,pp].T
            r = np.sign(np.array(w*x))
            deter_arg = np.argwhere((r[0]==y) == False)
            deter_arglist = [deter_arg[i][0] for i in range(len(deter_arg))]
            mm = len(deter_arglist)
            if len(deter_arglist) < mistake:
                w_best = copy.deepcopy(w)
                mistake = len(deter_arglist)
            loop += 1
            record.append(len(deter_arglist))
            record_best.append(mistake)
        return w_best.tolist()[0], 1- (mistake/size),record,record_best
def LogitRegression(data,x_len,y_col):
    size = len(data)
    data = np.array(data)
    x0 = np.mat([1 for i in range(size)]).T
    x = np.hstack((x0,data[:,0:x_len])).T
    y = data[:,y_col-1].reshape(1,size)
    w = np.zeros([1,x_len+1])
    
    epoch = 0
    record = []
    costlist = []
    while epoch < 7000:
        r = predit(x,w)
        costlist.append(cost(x,y,w))
        gradient1 = gradient(x,y,w)
        w -= gradient1.T
        epoch += 1
    
        r = np.sign(r - 0.5)
        
        mistake = np.sum(np.equal(r, y)==False)
        record.append(mistake)
    accuracy = 1-mistake/size
        
    return w.tolist()[0], record, costlist
def LinearRegression(data,x_len,y_col):
    size = len(data)
    data = np.array(data)
    x0 = np.mat([1 for i in range(size)]).T
    x = np.hstack((x0,data[:,0:x_len])).T
    y = data[:,y_col-1].reshape(size,1)

    w = (x*x.T).I * x * y
    error = np.sum(np.array(w.T * x - y.T).reshape(size,) ** 2)/size

    return w.tolist(), error
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

data_reg = []                                               # obtain data for linear regression
with open('linear-regression.txt','r') as file:
    line = file.readline().strip('\n').split(',')
    while line != ['']:
        line = list(map(float, line))
        data_reg.append(line)
        line = file.readline().strip('\n').split(',')
datareg_np = np.array(data_reg)

""" w, accuracy = PLA(data_clas,3,4)
print("The result weight(w0~wn) of PLA is: " + str(w) + " with accuracy: " + str(accuracy) + "%") """

w, accuracy,record,record_best = PLA(data_clas,3,5,"Pocket")

""" w, record, cost = LogitRegression(data_clas,3,5)
print("The result weight(w0~wn) of Logistic Regression is: " + str(w) + " with accuracy: " + str(1-record[-1]/20) + "%")

w, error = LinearRegression(data_reg,2,3)
print("The result weight(w0~wn) of Logistic Regression is: " + str(w)) """

""" plt.plot(np.arange(1,1000,1),record[1:1000],label='# of mistake points',color='r')
#plt.plot(np.arange(1,1200,1),record_best[1:1200],label='Pocket',color='b')
#plt.plot(np.arange(1,1000,1),cost[1:1000],label='cost',color='b')
plt.title('# of misclasified points vs iterations')
#plt.title('cost vs iterations')
plt.ylabel('# of misclasified points')
#plt.ylabel('Cost')
plt.xlabel('# of iterations')
plt.legend()
plt.show() """






dcla_po = []
dcla_ne = []
for row in data_clas:
    if row[4] == 1:               # row[3] for question 1, row[4] for question 2 3
        dcla_po.append(row[0:3])
    else:
        dcla_ne.append(row[0:3])
dcla_po = np.array(dcla_po)
dcla_ne = np.array(dcla_ne)

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})

#ax.scatter(datareg_np[:,0],datareg_np[:,1],datareg_np[:,2], c='r',marker='.')           #plot linear regression
ax.scatter(dcla_po[:,0],dcla_po[:,1],dcla_po[:,2], c='r',marker='+')
ax.scatter(dcla_ne[:,0],dcla_ne[:,1],dcla_ne[:,2], c='b',marker='_')



X = np.arange(0,1,0.1)
Y = np.arange(0,1,0.1)
X, Y = np.meshgrid(X,Y)
#Z = w[1] * X + w[2] * Y + w[0]
Z = -1 * (w[1]/w[3]) * X - (w[2]/w[3]) * Y -(w[0]/w[3])

surf = ax.plot_surface(X, Y, Z, cmap=cm.Blues,linewidth=0, antialiased=False)


plt.show() 


a = 1