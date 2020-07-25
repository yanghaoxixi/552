import numpy as np 




a = [[2.5,1.5],[1.5,-1.5]]

a = np.mat(a)
b,c = np.linalg.eig(a)



print(np.linalg.eig(a))