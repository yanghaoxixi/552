from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import math

# ------  Group member: Hao Yang, USCID 5284267375  --------

def Norm2(a,b):   # The Euclidean distance between point a and b
    sum = 0
    for i in range(len(a)):
        sum += (a[i] - b[i]) ** 2 
    return sum ** 0.5
def Kmeans(x, k):                                        # K-means algorithm, return [centroids, times of loop, points classification]
    dim = len(x[0])
    xx = []
    for j in range(dim):                                 # get list vector [x1,x2,x3....]
        xx.append([])
        for i in range(len(x)):
            xx[j].append(x[i][j])
    mins = []
    maxs = []
    for i in range(len(xx)):                             # get range of points
        mins.append(min(xx[i]))
        maxs.append(max(xx[i]))   
    centroids = []
    for o in range(k):                                   # step 1: create k random centroids
        centroids.append([])
        for p in range(dim):
            centroids[o].append(random.uniform(mins[p],maxs[p]))
    orig_cen = []
    loop = 0
    while sorted(centroids) != sorted(orig_cen) and loop <= 50000:     # loop until centroids will not change or loop more than 50000 times
        orig_cen = copy.deepcopy(centroids)
        group = []
        for g in range(k):
            group.append([])
        for row in x:                                       # step 2: assign all x to closest centroid
            dist = [Norm2(row,centroids[i]) for i in range(k)]
            posi = dist.index(min(dist))
            group[posi].append(x.index(row))
        if [] in group:                                     # if have empty group, should restart with better initial centroids 
            for o in range(k):                                   
                for p in range(dim):
                    centroids[o][p] = random.uniform(mins[p],maxs[p])
            continue
        for clas in range(len(group)):                      # step 3: recompute centroids
            if len(group[clas]) != 0:    
                for v in range(dim):
                    summ = 0
                    for ele in group[clas]:
                        summ += x[ele][v]
                    centroids[clas][v] = summ/len(group[clas])
        loop += 1
    return [centroids, loop, group]
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
    para = []                           # Step 1: Initially assign μ, Σ, α
    ran = [int(random.uniform(0,len(x))) for i in range(k)]
    for i in range(k):
        para.append([])
        para[i].append(x[ran[i]])
        z = mat(zeros((dim,dim)))
        z1 = [random.uniform(0,1) for i in range(dim)]
        z2 = mat(diag(z1))
        para[i].append(z + z2)
        para[i].append(1/k)
    x = np.array(x).T
    x = x.tolist()
    xm = np.mat(x)
    """ for ab in range(len(x[0])):    # Step 1: Initially assign ric (use if start with initialize ric)  
        a = [random.random() for i in range(k)]
        a = a/sum(a)
        a = a.tolist()
        r.append(a) """
    loop = 0
    LLD_dif = 1000
    LLD = 0
    while (loop <= 200 and LLD_dif > 1 * 10 ** -5) or LLD_dif<0:                      # loop 200 times
        r = []
        LLD_new = 0
        for i in range(len(x[0])):                  # Step 2: figure out ric
            r.append([])
            for j in range(k):
                r[i].append(Gau_mix_dtb(xm[:,i],mat(para[j][0]),mat(para[j][1]),para[j][2],j))
            bot = sum(r[i])
            LLD_new += math.log(bot)
            r[i] = [r[i][o]/bot for o in range(k)]      
        group = []                          # classificate x into k groups
        for grou in range(k):
            group.append([])
        for i in range(len(r)):                                    
            loc = r[i].index(max(r[i]))
            group[loc].append(i)
        para = []                           # Step 3: recompute μ, Σ, α with ric
        for i in range(k):                  # para = [[μ1,Σ1,α1],[μ2,Σ2,α2],[μ3,Σ3,α3]]
            para.append([])
            miuc_now = miuc(r,x,i)                              # miu
            para[i].append(miuc_now)                            # miu
            para[i].append(Ec(r,x,miuc_now,i))                  # E
            para[i].append(arfc(r,i))                           # arf
        loop += 1
        LLD_dif = LLD_new-LLD
        LLD = LLD_new
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

[cen,loop1,group1] = Kmeans(x,3)                  # print result of K-means
print('The result of clustering with K-means(loop and centroids):')
print(cen)
print(loop1)
[pa, loop2, group2] = GMM(x,3)
print('The result of clustering with GMM(loop and parameters):')
print("final loop time:" + str(loop2))
print(pa)


# below is for print the image of result
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