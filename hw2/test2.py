from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import math


def Gau_mix_dtb(xi,miu,E,arf,c):                         # ric : possibility of xi belong to group c (without normalization)
    dim = len(xi.tolist())
    a =  (2 * pi) ** (dim/2)
    b = (np.linalg.det(E)) ** 0.5
    ab = (a * b) ** -1
    c = xi - miu
    dd = c.T * E.I * c
    d = dd * (-1/2)
    px = ab * (e ** d.tolist()[0][0])
    top = arf * px
    return top
def miuc(r,x,c):                          # compute μ of group c, use x,ric
    top = []
    for i in range(len(x)):
        top.append(float(0))
    top = mat(top)
    x = np.array(x).T.tolist()
    xm = mat(x)
    bot = float(0)
    for i in range(len(x)):
        top += xm[i,:] * r[i][c]
        bot += r[i][c]
    miuc = (top/bot).T.tolist()

    return miuc
def Ec(r,x,miuc,c):                       # compute Σ of group c, use x,ric,μc
    dim = len(x)
    top = mat(zeros((dim,dim)))
    xm = mat(x)
    bot = float(0)
    miuc = mat(miuc)
    for i in range(len(x[0])):
        pls = (xm[:,i] - miuc) * (xm[:,i] - miuc).T
        top += r[i][c] * pls
        bot += r[i][c]
    Ec = (top/bot).tolist()
    return Ec
def arfc(r,c):                            # compute α of group c, use ric
    bot = 0
    top = 0
    for i in range(len(r)):
        top += r[i][c]
        bot += 1
    arfc = top/bot
    return arfc


def GMM(x,k):                            # K-means algorithm, return μ, Σ, α(Π in class)
    dim = len(x[0])
    
    para = []
    ran = [int(random.uniform(0,len(x))) for i in range(k)]
    for i in range(k):
        para.append([])
        para[i].append(x[ran[i]])
        z = mat(zeros((dim,dim)))
        z1 = [0.1 for i in range(dim)]
        z2 = mat(diag(z1))
        para[i].append(z + z2)
        para[i].append(1/k)
    
    x = np.array(x).T
    x = x.tolist()
    xm = np.mat(x)
    
    
    
    r = []                              # Step 1: Initially assign ric
    for i in range(len(x[0])):
        r.append([])
        for j in range(k):
            r[i].append(Gau_mix_dtb(xm[:,i],mat(para[j][0]),mat(para[j][1]),para[j][2],j))
        bot = sum(r[i])
        r[i] = [r[i][o]/bot for o in range(k)]
    
    loop = 0
    LLD_dif = 1000
    LLD = 0
    while (loop <= 200 and LLD_dif > 1 * 10 ** -5) or LLD_dif<0:                      # loop 100 times
        group = []                          # classificate x into k groups
        for grou in range(k):
            group.append([])
        for i in range(len(r)):                                    
            loc = r[i].index(max(r[i]))
            group[loc].append(i)
        para = []                           # Step 2: figure out μ, Σ, α with ric
        for i in range(k):                  # para = [[μ1,Σ1,α1],[μ2,Σ2,α2],[μ3,Σ3,α3]]
            para.append([])
            miuc_now = miuc(r,x,i)                              # miu
            para[i].append(miuc_now)                            # miu
            para[i].append(Ec(r,x,miuc_now,i))                  # E
            para[i].append(arfc(r,i))                                  # arf
        r_new = []                          # Step 3: recompute ric
        LLD_new = 0
        for i in range(len(x[0])):
            r_new.append([])
            for j in range(k):
                r_new[i].append(Gau_mix_dtb(xm[:,i],mat(para[j][0]),mat(para[j][1]),para[j][2],j))
            bot = sum(r_new[i])
            LLD_new += math.log(bot)
            r_new[i] = [r_new[i][o]/bot for o in range(k)]
            r = copy.deepcopy(r_new)
    
        loop += 1
        if loop//20 == loop/20:
            print(loop)
        LLD_dif = LLD_new-LLD
        LLD = LLD_new
        print(LLD_dif)
    print("final loop time:" + str(loop))
    print("final LLD_dif:" + str(LLD_dif))
    return [para, loop, group]



x1 = []
x2 = []
x = []
with open('clusters.txt','r') as file:              #get coordinate x,y(list of float)
    line = file.readline()
    while line:
        line = line.strip('\n').split(',')
        x1.append(float(line[0]))
        x2.append(float(line[1]))
        x.append([x1[-1],x2[-1]])
        line = file.readline()
mins = [min(x1),min(x2)]
maxs = [max(x1),max(x2)]                            # range of all points


[pa, loop2, group2] = GMM(x,3)



cen = []
for i in range(3):
    cen.append([])
    for j in range(2):
        cen[i].append(pa[i][0][j][0])

c1 = [cen[i][0] for i in range(len(cen))]
c2 = [cen[i][1] for i in range(len(cen))]
x11 = [x[i][0] for i in group2[0]]
x12 = [x[i][1] for i in group2[0]]
x21 = [x[i][0] for i in group2[1]]
x22 = [x[i][1] for i in group2[1]]
x31 = [x[i][0] for i in group2[2]]
x32 = [x[i][1] for i in group2[2]]
plt.plot(x11,x12,'bo')
plt.plot(x21,x22,'g^')
plt.plot(x31,x32,'kD')
plt.plot(c1,c2,'r+')
plt.show()



a = 1