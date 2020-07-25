import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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

pca = PCA(n_components=2)
ve = pca.fit(datanp)
print(pca.explained_variance_ratio_)
reduced_x=pca.fit_transform(datanp)     #dimen reduction

plt.plot(reduced_x[:,0],reduced_x[:,1],'r.')
plt.show()

for i in range(1,5):
    print(i)
a = 1