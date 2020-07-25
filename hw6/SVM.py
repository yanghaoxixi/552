import numpy as np 
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

# ----------- Hao Yang, USCID:5284267375 ------------------------------

def SVM(data,kernel=False):
    x = np.mat(data[:,0:2])
    y = np.mat(data[:,2].reshape(100,1))
    xx = x*x.T
    if kernel==True:
        #xx = xx + 1
        xx = np.multiply(xx,xx)
        
    yy = y*y.T
    P = matrix(np.multiply(xx,yy))# P
    q = matrix(np.mat([-1 for i in range(100)]).T,tc='d')    # q
    GG = -1 * np.eye(100)
    G = matrix(GG)                          # G
    h = matrix([0 for i in range(100)],tc='d')     # h
    A = matrix(y.T)                         # A
    b = matrix([0],tc='d')                                   # b

    result = solvers.qp(P,q,G,h,A,b)
    arf = np.mat(result['x'])
    
    if kernel == False:
        ya = np.multiply(y,arf)
        w = np.multiply(ya,x).sum(axis=0)           # w



        o = np.argwhere(arf == max(arf)).tolist()[0][0]
        b = 1/y[o] - w*x[o,:].T                     #b

        sv = np.argwhere(arf > 1)[:,0].tolist()

        """ arf = np.array(arf).reshape(100,)
        top2 = np.argsort(arf)[::-1]
        arf_sort = np.sort(arf)[::-1] """

        return w.tolist()[0],b.tolist()[0][0],sv
    else:
        """ arf = np.array(arf).reshape(100,)
        arf_sort = np.sort(arf)[::-1] """
        sv = np.argwhere(arf > 1e-5)[:,0].tolist()
        return sv
    


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



w,b,sv = SVM(linnp)
sv2 = SVM(nonnp,kernel=True)

print('the weight is %s and bias is %f' % (w,b))
print('The equation of the classification line:')
print('Y=%fX%f' % (-w[0]/w[1],b))
print('Support vectors:\nindex:'+str(sv)+'\ncoordinates:')
print(linnp[sv][:,0:2])

print('Support vectors for non seperable data:\nindex:'+str(sv2)+'\ncoordinates:')
print(nonnp[sv2][:,0:2])
# -------------------  plot ----------------------------------

""" lin_po = []
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

svp = nonnp[sv2]
plt.scatter(svp[:,0],svp[:,1],c='b',marker='.')

X = np.linspace(0,1)
Y = -w[0]/w[1] * X + b
plt.plot(X,Y)

plt.xlim(0,1)
plt.ylim(0,1)
plt.show() """

""" dis = []
for i in sv[0:2]:
    dis.append( abs( (w[0]*linnp[i][0]+w[1]*linnp[i][1]-w[1]*b)/((w[0]**2 + w[1]**2)**0.5) ) ) """

#print(d)



a = 1