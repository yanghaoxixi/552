import matplotlib.pyplot as plt
fig,ax=plt.subplots()
import numpy as np

data_clas = []                                              # obtain data for classification
with open('classification.txt','r') as file:
    line = file.readline().strip('\n').split(',')
    while line != ['']:
        line = list(map(float, line))
        data_clas.append(line)
        line = file.readline().strip('\n').split(',')
dataclas_np = np.array(data_clas)  

X=dataclas_np[:,[0,2]]  #提取特征
y=dataclas_np[3]   #提取目标

###把数据归一化###
mean=X.mean(axis=0)
sigma=X.std(axis=0)
X=(X-mean)/sigma

###提取不同类别的数据，用于画图###
x_positive=X[y==1]
x_negative=X[y==-1]

ax.scatter(x_positive[0],x_positive[1],marker="o",label="y=+1")
ax.scatter(x_negative[0],x_negative[1],marker="x",label="y=-1")
ax.legend()
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title("standardized Data")

X[2]=np.ones((X.shape[0],1))   #增加一列常数项
X=X.values    #把特征转换成ndarray格式

###初始化w###
w=X[0].copy()  #选取原点到第一个点的向量作为w的初始值
w[2]=0  #增加一项---阈值，阈值初始化为0
w=w.reshape(3,1)

y=y.values.reshape(100,1)  #把目标转换成ndarray格式，形状和预测目标相同
    
def compare(X,w,y):
    ###用于比较预测目标y_pred和实际目标y是否相符，返回分类错误的地方loc_wrong###
    ###输入特征，权重，目标###
    scores=np.dot(X,w)  #把特征和权重点乘，得到此参数下预测出的目标分数
    
    y_pred=np.ones((scores.shape[0],1))  #设置预测目标，初始化值全为1，形状和目标分数相同
    
    loc_negative=np.where(scores<0)[0]  #标记分数为负数的地方
    y_pred[loc_negative]=-1  #使标记为负数的地方预测目标变为-1
    
    loc_wrong=np.where(y_pred!=y)[0]  #标记分类错误的地方
    
    return loc_wrong

def update(X,w,y):
    ###用于更新权重w，返回更新后的权重w###
    ###输入特征，权重，目标###
    num=len(compare(X,w,y)) #分类错误点的个数
    w=w+y[compare(X,w,y)][np.random.choice(num)]*X[compare(X,w,y),:][np.random.choice(num)].reshape(3,1)
    return w

def perceptron_pocket(X,w,y):
    ###感知机口袋算法，显示n次迭代后最好的权重和分类直线，并画出分类直线###
    ###输入特征，初始权重，目标###
    best_len=len(compare(X,w,y))  #初始化最少的分类错误点个数
    best_w=w  #初始化口袋里最好的参数w
    for i in range(100):
        print("错误分类点有{}个。".format(len(compare(X,w,y))))
        w=update(X,w,y)
        #如果当前参数下分类错误点个数小于最少的分类错误点个数，那么更新最少的分类错误点个数和口袋里最好的参数w
        if len(compare(X,w,y))<best_len:
            best_len=len(compare(X,w,y))
            best_w=w

    print("参数best_w:{}".format(best_w))
    print("分类直线:{}x1+{}x2+{}=0".format(best_w[0][0],best_w[1][0],best_w[2][0]))
    print("最少分类错误点的个数:{}个".format(best_len))
    line_x=np.linspace(-3,3,10)
    line_y=(-best_w[2]-best_w[0]*line_x)/best_w[1]
    ax.plot(line_x,line_y)

plt.show()