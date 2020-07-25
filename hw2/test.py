from numpy import *
import numpy as np
import random
import math
from sklearn import mixture
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
from sklearn.cluster import KMeans
from matplotlib.patches import Ellipse


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse (椭圆)with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))

def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')

    w_factor = 0.2 / gmm.weights_.max()
    i=0
    for  pos, covar, w in zip(gmm.means_, gmm.precisions_, gmm.weights_):
        if i<=3:
            draw_ellipse(pos, covar, alpha=w * w_factor)
            i=i+1



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
x = np.array(x)

X, y_true = make_blobs(n_samples=400, centers=3, cluster_std=0.60, random_state=0)
#X = X[:, ::-1] # flip axes for better plotting
gmm = mixture.GaussianMixture(n_components=3)
""" kmeanss = KMeans(n_clusters=3).fit(x) """
""" labels = gmm.predict(x) """
plot_gmm(gmm,x)
""" plt.scatter(x[:, 0], x[:, 1], c=labels, s=50, cmap='viridis') """
plt.show()


a = 1