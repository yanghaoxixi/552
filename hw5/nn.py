import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt

#--------- Hao Yang, USCID:5284267375 -------------------------
def Convert_data(data_train):    # get dataset
    imgfile = "./"+data_train[0]
    img = np.array(Image.open(imgfile).convert('L'), 'f')
    m,n = list(img.shape)        # size of image
    x = (img/255).reshape(1,m*n).T
    y = [1] if "down" in imgfile else [0]
    for i in range(1,len(data_train)):
        imgfile = "./"+data_train[i]
        # size of image is 30 * 32 = 960, convert into vector of 1 * 960 
        img = (np.array(Image.open(imgfile).convert('L'), 'f')/255).reshape(1,m*n).T
        x = np.hstack((x,img))
        y.append(1 if "down" in imgfile else 0)
    y = np.array(y).T
    return x,y
def sigmoid(s):         
    return np.exp(s)/(1 + np.exp(s))
def FNN(x,y,lr=0.1,epoch=1000,hl=100,ini=0.01):
    input_size = list(x.shape)[0]           # get input size
    image_num = list(x.shape)[1]
    # initialize weight and bias from input layer to hidden layer
    v = np.mat([random.uniform(-ini,ini) for i in range(input_size)])
    r = np.mat(random.uniform(-ini,ini))
    for nerual in range(1,hl):
        v = np.vstack((v,np.mat([random.uniform(-ini,ini) for i in range(input_size)])))
        r = np.vstack((r,random.uniform(-ini,ini)))
    # initialize weight and bias from hidden layer to output layer
    w = np.mat([random.uniform(-ini,ini) for i in range(hl)])
    o = random.uniform(-ini,ini)
    # ----------- finished initialization --------------------
    loss = []
    for loop in range(epoch):
        print(str(loop)+'/1000')
        cost = []
        for item in range(image_num):
            b = sigmoid(v*(x[:,item].reshape(960,1)) - r).tolist()
            y_pre = (sigmoid(w * sigmoid(v*(x[:,item].reshape(960,1)) - r) - o)).tolist()[0][0] 
            Ek = 0.5 * ((y_pre - y[item])**2)
            cost.append(Ek)
            # start computing gradients
            g = y_pre * (1-y_pre) * (y[item]-y_pre)
            eh = []
            for h in range(hl):
                bh = b[h][0]
                eh.append(bh*(1-bh)*g*(w.tolist()[0][h]))
            # update weights and biases
            w += lr*g*sigmoid(v*(x[:,item].reshape(960,1)) - r).T
            o -= lr * g
            eh = np.mat(eh).reshape(1,100)
            v += (eh.T * x[:,item].reshape(960,1).T)*lr
            r -= lr*eh.T
        loss.append(sum(cost))
    return v,r,w,o,loss

def FNN_test(x,y,v,r,w,o):
    image_num = list(x.shape)[1]
    y_pre = []
    for item in range(image_num):
        pre = (sigmoid(w * sigmoid(v*(x[:,item].reshape(960,1)) - r) - o)).tolist()[0][0]
        result = 1 if pre>=0.5 else 0
        y_pre.append(result)
    y_pre = np.array(y_pre)
    acc = np.sum(np.equal(y_pre, y)==True)
    
    
    return y_pre.tolist(), acc/image_num
    


data_train = []
with open('downgesture_train.list','r') as train:
    line = train.readline()
    while line != '':
        
        data_train.append(line.strip('\n'))
        line = train.readline()
data_test = []
with open('downgesture_test.list','r') as test:
    line = test.readline()
    while line != '':
        
        data_test.append(line.strip('\n'))
        line = test.readline()

x,y = Convert_data(data_train)
x_test,y_test = Convert_data(data_test)



v,r,w,o,loss = FNN(x,y)
prediction, acc = FNN_test(x_test,y_test,v,r,w,o)
print("Prediction of test data:" + str(prediction))
print("Accuracy rate:" + str(acc))

plt.plot(loss)
plt.show()

a = 1