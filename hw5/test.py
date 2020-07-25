import numpy as np 
import random
from sklearn.neural_network import MLPClassifier
from PIL import Image
import matplotlib.pyplot as plt

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
    
    return x.T,y



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


imgfile = "./"+data_test[15]
img = Image.open(imgfile).convert('L').show()


clf = MLPClassifier(solver='sgd',activation='logistic',hidden_layer_sizes=(100),learning_rate_init=0.1,max_iter=1000)
clf.fit(x,y)
y_pred = clf.predict(x_test)

acc = (np.sum(np.equal(y_pred, y_test)==True))/y_pred.shape

print(y_pred)
print(acc)





a = 1