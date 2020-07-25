import numpy as np
import itertools as it

def get_18_from_trible(trible):  #把三维向量变为代表特征的18维向量
    a = np.zeros([2,3,3])
    # print(a)
    # print(trible)
    a1 = trible[0]
    a2 = trible[1]
    a3 = trible[2]
    print(a3)
    if a1 == 3:  #对于第一种属性为"*"的情况
        a1 = [1,2]
    else:
        a1 = [a1]    
    if a2 == 4:  #对于第二种属性为"*"的情况
        a2 = [1,2,3]
    else:
        a2 = [a2]
    if a3 == 4:  #对于第三种属性为"*"的情况
        a3 = [1,2,3]
    else:
        a3 = [a3]
    # print (a1,a2,a3) 
    print('sdasdasdasd')
    print(a3)
    q = 1
    for m1 in a1:
        for m2 in a2:
            for m3 in a3:
                a[m1-1][m2-1][m3-1] = 1
    return a           #得到了一个18维向量（0/1二值），代表18种特征情况

def turn_48_to_trible(num):   
# num in [0,47]，把一个小于48的数字对应到一个三维数组中
    # print (num)
    for i in range(3): 
        for j in range(4):
            for k in range(4):
                if i*16 + j*4 + k == num:
                    return [i+1,j+1,k+1]

def from_48_to_18(num):  #把0-47的某个数唯一对应到某个18维向量
    a = turn_48_to_trible(num)
    b = get_18_from_trible(a)
    return b  

def main(k):
    rset=[]
    for i in it.combinations(range(48),k):   
    #开始对48取k的组合数进行穷举，i是一个k元数组
        subset=[]
        for j in range(k): 
            p = from_48_to_18(i[j])  
            subset.append(p)
        subset = np.array(subset)    
        subset = subset.any(axis=0)  # 这是去除冗余操作！！！
        subset = np.reshape(subset,[18]) 
        subset = subset.tolist()  #从array变为list方便接下来的操作
        count = 0
        for i in range(18):
            count += 2 ** i *subset[i]  
            #这是简单的一一映射，18维二值向量可以一一对应到1~2^18的某个数
        subset = count  
        length = len(rset)
        rset.append(subset)    
        if len(rset) % 100000 == 0 :
            print(len(rset))  #为了证明程序确实在运行hhh，这个可以注释掉
        if len(rset) > 5000000: 
            rset = list(set(rset))  #set是集合，是对rset数组进行去重！
            #设置500W上限为了防止set操作时数组长度太长导致程序崩掉
    rset = list(set(rset)) #最终set操作一下得到最终结果
    #rset = list(set(tuple(t) for t in rset))

    print( "%d ： %d examples"%(k,len(rset)))    

main(2) # 运行即得k个析取范式的结果。注：当k≥9时花费时间过长！目前没法解决！

a = 1