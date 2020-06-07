# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 20:27:24 2020

@author: Gavin
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import datasets

digits = datasets.load_digits()

print(digits.data.shape)

#normalize values to [0,1]
X = digits.data / 255.


plt.gray() 

for i in range(10):
    plt.figure()
    plt.matshow(digits.data[i].reshape(8, 8))
    plt.show()
    

pca = PCA(n_components=32)
X_transformed = pca.fit_transform(X)

cumsum_var=np.cumsum(pca.explained_variance_ratio_)
print(cumsum_var)

plt.figure()
plt.title('Cumulative Explained Variance [A]')
plt.xlabel('Dimentions')
plt.ylabel('Cumulative Variance')
plt.plot(range(0,32), cumsum_var)

#now let's look at some images, reconstructed after only using 
# n_components PCA dimensions
"""
X_reproduce = pca.inverse_transform(X_transformed)

for i in range(10):
    plt.figure()
    plt.matshow(X[i].reshape(8, 8)) 
    plt.title('Original image sample # = %i'%(i))
    plt.show()
    plt.figure()
    plt.matshow(X_reproduce[i].reshape(8, 8))
    plt.title('Reconstructed image sample # = %i'%(i))
    plt.show()
    
"""

pca = PCA(n_components=2)
X_transformed = pca.fit_transform(X)

X_reproduce = pca.inverse_transform(X_transformed)

for i in range(5):
    plt.figure()
    plt.matshow(X[i].reshape(8, 8)) 
    plt.title('N Component=2: Original IMG # = %i'%(i))
    plt.show()
    plt.figure()
    plt.matshow(X_reproduce[i].reshape(8, 8))
    plt.title('N Component=2: Reconstructed IMG # = %i'%(i))
    plt.show()