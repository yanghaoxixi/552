import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

# Group member: Hao Yang USCID: 5284267375

def PCA(x, k):                                          # PCA function, reduce dimension of x to k
    dim = len(x[0])
    size = len(x)
    xnp = np.array(x)
    
    med = (xnp.sum(axis=0))/size                        # Step 0: centralization
    xnp = xnp - med
    x = xnp.tolist()
    
    xm = np.mat(xnp).T                                  # Step 1: compute covariance matrix E
    E = xm * xm.T
    E = E/size

    eigval, feavec = np.linalg.eig(E)                   # Step 2: compute eigenvalues and featurevectors of E    
    idx = eigval.argsort()[::-1]                        # in decreasing order
    ev = eigval[idx[0]]
    fv = feavec[:,idx[0]]
    for i in idx[1:]:
        fv = np.c_[fv,feavec[:,i]]
        ev = np.append(ev,eigval[i])
    
    Utr = fv[:,0:k]                                   # Step 3: got U truncate
    origin_direction = np.eye(dim)
    new_direction = Utr.T * origin_direction
    z = Utr.T *xm

    e = (sum(ev[:k])/sum(ev)) * 100

    return Utr, new_direction, z, e
def Fastmap(wl,wd,k):                                   # Fastmap function, use wordlist and worddistance,embeded in k dimension
    size = len(wl)
    coordinate = [[] for i in range(size)]
    
    for loop in range(k):                                             # Embeded in k dimension
        
        pointa, pointb, dis = Farthest(wd,size,loop,coordinate)       # Step 1: find farthest pair Oa,Ob in O(10N) time
        for i in range(1,size+1):                                     # Step 2: consider all other objects one by one, and get their next coordinate in new space.  in O(N) time
            if i == pointa:
                coordinate[i-1].append(0)
            elif i == pointb:
                coordinate[i-1].append(dis)
            else:
                coordinate[i-1].append(xiCompute(wd,pointa,pointb,i,loop,coordinate))

    return coordinate
def xiCompute(wd,Oa,Ob,Oi,loop,x):                                 # compute xi of Oi based on Oa---Ob
    Dab = NowDistance(wd,Oa,Ob,loop,x)
    if Oa < Oi:
        Dai = NowDistance(wd,Oa,Oi,loop,x)
    else:
        Dai = NowDistance(wd,Oi,Oa,loop,x)
    if Ob < Oi:
        Dbi = NowDistance(wd,Ob,Oi,loop,x)
    else:
        Dbi = NowDistance(wd,Oi,Ob,loop,x)
    xi = (Dab**2 + Dai**2 - Dbi**2)/(2 * Dab)
    return xi
def Farthest(wd,size,loop,x):
    IterationMax = 5                                        # set the max iteration time as 5, avoid more than 2 points have same farthest dist and cant stop.
    Oa = random.randint(1,size)                            # start from random point, find farthest point of it
    Ob = Farthestpoint(wd,size,Oa,loop,x)                        # find all distace from point "Oa". Takes O(N) time
    for i in range(IterationMax):                           #5 iteration max   O(10N) max
        O_next = Farthestpoint(wd,size,Ob,loop,x)
        if O_next == Oa:
            break
        Oa = O_next

        O_next = Farthestpoint(wd,size,Oa,loop,x)
        if O_next == Ob:
            break
        Ob = O_next
    if Oa > Ob:
        return Ob,Oa,NowDistance(wd,Ob,Oa,loop,x)
    else:
        return Oa,Ob,NowDistance(wd,Oa,Ob,loop,x)
def Farthestpoint(wd,size,pointA,loop,x):                          # find farthest point from A
    max = -1
    for i in range(1,size+1):                               # O(N)      
        if i < pointA:
             if NowDistance(wd,i,pointA,loop,x) > max:
                 pointB = i
                 max = NowDistance(wd,i,pointA,loop,x)
        elif i > pointA:
            if NowDistance(wd,pointA,i,loop,x) > max:
                pointB = i
                max = NowDistance(wd,pointA,i,loop,x)
    return pointB
def NowDistance(wd,Oa,Ob,loop,x):                               # Dnew in this loop 
    dis = wd[Oa,Ob]
    for i in range(loop):
        dis = (dis**2 - (x[Oa-1][i]-x[Ob-1][i])**2)**(1/2)
    if type(dis) == complex:
        return 0
    else:
        return dis

data_pca = []
with open('pca-data.txt', 'r') as file1:                # got dataset for implement PCA
    line = file1.readline().strip('\n').split('\t')
    line = list(map(float, line))
    data_pca.append(line)
    line = file1.readline().strip('\n').split('\t')
    while line != ['']:
        line = list(map(float, line))
        data_pca.append(line)
        line = file1.readline().strip('\n').split('\t')        
datanp = np.array(data_pca)
k_eigenvector, dir, z, rate = PCA(data_pca,2)
print("the first k eigenvector is: ")
print(k_eigenvector)
print("the coordinate system of 2D spcae in 3D space is:" + str(dir))

wordlist = []
distance = {}
with open('fastmap-wordlist.txt','r') as wl:            # got wordlist
    line = wl.readline().strip('\n')
    while line:
        wordlist.append(line)
        line = wl.readline().strip('\n')
with open('fastmap-data.txt','r') as wd:                # got distance data
    line = wd.readline().strip('\n').split('\t')
    while line != ['']:
        distance[int(line[0]),int(line[1])] = float(line[2])
        line = wd.readline().strip('\n').split('\t')

coordinates =  Fastmap(wordlist,distance,2)
coordinates = np.array(coordinates)

plt.scatter(coordinates[:,0],coordinates[:,1],marker='.',color='r',label='1')
for i in range(len(wordlist)):
    plt.annotate(wordlist[i],xy = (coordinates[:,0][i],coordinates[:,1][i]), xytext = (coordinates[:,0][i]-0.1,coordinates[:,1][i]+0.1))
""" plt.xlim(-1,13)
plt.ylim(-1,13) """
plt.show()


# ------------------ here is the improvement part to check the information remain rate --------------------------
""" higher_space_check =  Fastmap(wordlist,distance,4)
q, w, e, information_rate = PCA(higher_space_check,2)
print("The information contain rate in 2D space is:")
print(information_rate) """


a = 1



#----------------- draw image for pca below --------------------------
""" start = np.mat((datanp.sum(axis=0))/len(datanp))
end = start + dir * 40
end[1] = end[1]/2
both1 = np.r_[start,end[0]].T
both1 = both1.tolist()
both2 = np.r_[start,end[1]].T.tolist()


fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
ax.scatter(datanp[:,0],datanp[:,1],datanp[:,2], c='r',marker='.')

ax = fig.gca(projection='3d')
ax.plot(both1[0],both1[1],both1[2],c='b')
ax.plot(both2[0],both2[1],both2[2],c='b')
plt.show() 



plt.plot(z[0],z[1],'b.')
plt.show() """


a = 1