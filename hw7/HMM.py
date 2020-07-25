import numpy as np 
import math

#------------- Hao Yang, USCID: 5284267375 --------------------------

def distances(position):   # position: (x,y)  [0,9]   x-row   y-column
    if len(position) != 2 or max(position)>9 or min(position)<0:
        return 'error'
    d = []
    d.append((position[0] ** 2 + position[1] ** 2) ** 0.5)
    d.append((position[0] ** 2 + (9 - position[1]) ** 2) ** 0.5)
    d.append(((9 - position[0]) ** 2 + position[1] ** 2) ** 0.5)
    d.append(((9 - position[0]) ** 2 + (9 - position[1]) ** 2) ** 0.5)
    
    return d
def gotPai(grid,obs,lenQ,size2):   # got pai
    P = np.zeros((1,lenQ))
    for i in range(lenQ):
        x = i//10
        y = i-x*10
        realDistance = distances((x,y))
        realRange = []
        for j in range(size2):
            realRange.append([math.ceil(7*realDistance[j])/10,math.floor(13*realDistance[j])/10])
        qqq = 0
        for j in range(size2):
            if obs[0][j]>=realRange[j][0] and obs[0][j]<=realRange[j][1]:
                qqq += 1
        if qqq == 4 and grid[x][y] == 1:
            P[0,i]+=1
    ppp = np.sum(P)
    for i in range(lenQ):
        if P[0,i] == 1:
            P[0,i] = 1/ppp
    return P
def gotA(grid,obs):   # got A
    size = len(grid) ** 2
    A = np.zeros((size,size))
    for i in range(10):
        for j in range(10):
            if grid[i][j] == 1:
                ppp = 0
                if i-1 >= 0 and grid[i-1][j]==1: # up
                    ppp+=1
                    A[i*10+j,i*10+j-10] = 1
                if i+1 <= 9 and grid[i+1][j]==1: # down
                    ppp+=1
                    A[i*10+j,i*10+j+10] = 1
                if j-1 >= 0 and grid[i][j-1]==1: # left
                    ppp+=1
                    A[i*10+j,i*10+j-1] = 1
                if j+1 <= 9 and grid[i][j+1]==1: # down
                    ppp+=1
                    A[i*10+j,i*10+j+1] = 1
                A[i*10+j,:] = A[i*10+j]/ppp
    return A
def gotB(grid,obs,lenV):     # got B
    row = len(grid)**2
    B = np.zeros((row,1))
    for i in range(row):
        x = i//10
        y = i-x*10
        realDistance = distances((x,y))
        realRange = []
        pro = 1
        for j in range(4):
            realRange.append([math.ceil(7*realDistance[j])/10,math.floor(13*realDistance[j])/10])    
            pro *= (realRange[j][1]-realRange[j][0]) * 10 + 1
        B[i,0] = 1/round(pro)
    return B
def ifIn(distanceList,i): # distanceList-the observation value, i- the index of point:0-99
    x = i//10
    y = i-x*10
    dist = distances((x,y))
    realRange = []
    for j in range(4):
        realRange.append([math.ceil(7*dist[j])/10,math.floor(13*dist[j])/10])
    for k in range(len(distanceList)):
        if distanceList[k]<realRange[k][0] or distanceList[k]>realRange[k][1]:
            return False
    return True
def gotABpai(grid,obs):
    size = len(grid)    # size of map grid : 10
    observeRange = 1.3 * ((size - 1) * (2**0.5))  # range of possible observe value: [0, 1.3 * max(d)], d = longest possible range
    lenV = int(observeRange * 10)  # lenV = 165. The set of possible observe value: [0,16.5] with one decimal place.  M = 165
    size2 = len(obs[0])  # size of observe value : 4
    lenQ = size ** 2   # N = 100
    #  so need a A with size of 100 x 100, B with size of 100 x 165**4, pai with size 100 * 1 .T
    P = gotPai(grid,obs,lenQ,size2)   # got P
    A = gotA(grid,obs)
    B = gotB(grid,obs,lenV)
    return P,A,B
def HMM(P,A,B,grid,obs):
    Step = len(obs) # len of observations, 11
    d = np.zeros((1,100))
    v = np.zeros((1,100))
    for i in range(100):
        if grid[i//10][i-(i//10)*10] == 1 and ifIn(obs[0],i):
            d[0,i] = (1/87) * B[i,0]
    for loop in range(1,Step):   # 11 time-step
        d_next = np.zeros((1,100))
        v2 = np.zeros((1,100))
        for i in range(100):
            record = np.multiply(d[loop-1,:],A[:,i].T)
            if ifIn(obs[loop],i):
                d_next[0,i] = record.max() * B[i,0]
                if record.max() != 0:
                    v2[0,i] = record.argmax()
        d = np.vstack((d,d_next))
        v = np.vstack((v,v2))

    trajectory = []
    trajectory.append(d[-1].argmax())
    for epoch in range(Step-1,0,-1):
        trajectory.append(int(v[epoch,trajectory[10-epoch]]))
    for i in range(len(trajectory)):
        x = trajectory[i]//10
        y = trajectory[i]- 10*x
        trajectory[i] = (x,y)
    return trajectory

grid = []
obs = []
with open("hmm-data.txt",'r') as file:
    file_lines = file.readlines()    
    for line in file_lines[2:12]:
        grid.append(list(map(float, line.split(' '))))
    for line2 in file_lines[24:35]:
        row = line2.strip('\n').split(' ')
        if '' in row:
            row.remove('')
        row = list(map(float,row))
        obs.append(list(map(float, row)))

P,A,B =  gotABpai(grid,obs)
trajectory = HMM(P,A,B,grid,obs)


print("The most likely trajectory of robot:")
for ele in trajectory[::-1]:
    print(ele)




a = 1