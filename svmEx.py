import numpy as np 
from pylab import *
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn import svm, datasets




def createClusteredData(N, k):
    np.random.seed(1234)
    pointsPerCluster = float(N)/k
    X = []
    y = []
    for i in range (k):
        incomeCentroid = np.random.uniform(20000.0, 200000.0)
        ageCentroid = np.random.uniform(20.0, 70.0)
        for j in range(int(pointsPerCluster)):
            X.append([np.random.normal(incomeCentroid, 10000.0), np.random.normal(ageCentroid, 2.0)])
            y.append(i)
            
    X = np.array(X)
    y = np.array(y)
    
    return X, y


(X, y) = createClusteredData(100, 5)

plt.figure(figsize=(8,6))
plt.scatter(X[:,0], X[:,1], c=y.astype(np.float))
plt.show()

scaling = MinMaxScaler(feature_range=(-1,1)).fit(X) #we've scaled our data for SVM to work more efficient
X = scaling.transform(X)

plt.figure(figsize=(8,6))
plt.scatter(X[:,0], X[:,1], c=y.astype(np.float))
plt.show()

C = 1.0
svc = svm.SVC(kernel='linear', C=C).fit(X,y)


    
def plotPredictions(clf):
    #we're creating dense grid of points to sample
    xx, yy = np.meshgrid(np.arange(-1, 1, .001),
                         np.arange(-1, 1, .001))
    
    #converting to numpy arr
    npx = xx.ravel()
    npy = yy.ravel()
    
    #converting to a list of 2D -income,ages- points
    samplePoints = np.c_[npx, npy]
    
    #labels for each point
    Z = clf.predict(samplePoints)
    
    plt.figure(figsize=(8,6))
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha =.8)
    plt.scatter(X[:,0], X[:,1], c = y.astype(np.float))
    plt.show()
 
    
plotPredictions(svc)

print(svc.predict(scaling.transform([[10000, 30]])))

    
