#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 21:16:21 2020

@author: Gavin Monroe
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import pandas as pd


# This code will suppress warnings.
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", ConvergenceWarning)

# load training set
ytrain = []
Xtrain = []
with open('HW3train.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',',lineterminator='\n')
    for row in reader:
        if len(row)==3:
            ytrain.append( int(row[0]) )
            Xtrain.append( [float(row[1]) , float(row[2]) ])
            
Xtrain = np.array(Xtrain)
ytrain = np.array(ytrain)

# load testing set
ytest = []
Xtest = []
with open('HW3test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',',lineterminator='\n')
    for row in reader:
        if len(row)==3:
            ytest.append( int(row[0]) )
            Xtest.append( [float(row[1]) , float(row[2]) ])
            
Xtest = np.array(Xtest)
ytest = np.array(ytest)


h = .03  # step size in the mesh
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh 
x1_min, x1_max = Xtrain[:, 0].min() - 1, Xtrain[:, 0].max() + 1
x2_min, x2_max = Xtrain[:, 1].min() - 1, Xtrain[:, 1].max() + 1
x1mesh, x2mesh = np.meshgrid(np.arange(x1_min, x1_max, h),
                         np.arange(x2_min, x2_max, h))

# Create color maps
cmap_light = ListedColormap(['lightblue', 'lightcoral', 'grey'])
cmap_bold = ListedColormap(['blue', 'red', 'black'])




alpharange = np.logspace(-6,0,5)
learnrateinitrange = np.logspace(-3,-1,3)
trainscore = np.zeros(10)
testscore = np.zeros(10)
nodes = np.zeros(10)
testnode = 0.0
testaccuracy = 0.0
testalpha = 0.0
testrate = 0.0
trainaccuracy = 0.0

maxtestacc = 0.0
maxtrainacc = 0.0

testalpha = 0.0
testrate = 0.0

trainalpha = 0.0
trainrate = 0.0

maxtrainalpha = 0.0
maxtrainrate = 0.0

maxtestalpha = 0.0
maxtestrate = 0.0

maxtrainnode = 0
maxtestnode = 0
i = 0
for value in range (1,11):
    nodetestacc = 0.0
    nodetrainacc = 0.0
    for a in alpharange:
        for rate in learnrateinitrange:
            clf = MLPClassifier(hidden_layer_sizes=value, activation='relu', solver='sgd', learning_rate = 'adaptive',alpha= a, learning_rate_init = rate, max_iter=200)
            clf.fit(Xtrain, ytrain)
            trainacc = clf.score(Xtrain, ytrain)
            testacc = clf.score(Xtest, ytest)
            if(testacc > nodetestacc):
                nodetestacc = testacc
                testscore[i] = testacc
                testalpha = a
                testrate = rate
            if(trainacc > nodetrainacc):
                nodetrainacc = trainacc
                trainscore[i] = trainacc
                trainalpha = a
                trainrate = rate
    nodes[i] = i
    if (maxtestacc < testscore[i]):
        maxtestacc = testscore[i]
        maxtestalpha = testalpha
        maxtestrate = testrate
        maxtestnode = i
        
    if (maxtrainacc < trainscore[i]):
        maxtrainacc = trainscore[i]
        maxtrainalpha = trainalpha
        maxtrainrate = trainrate
        maxtrainnode = i
    i = i + 1
    
       
              
print("The best test accuracy  "+ str(maxtestacc) + "with an alpha value pf " + str(maxtestalpha)+ ", a learning rate of " + str(maxtestrate) + ", and" + str(maxtestnode) + "nodes")
# train network with a single hidden layer of 2 nodes
plt.figure()
plt.title('Training Set of the number of hidden nodes')
plt.xlabel('Hidden Nodes')
plt.ylabel('Best Accuracy')
plt.plot(nodes, trainscore)

plt.figure()
plt.title('Test Set of the number of hidden nodes')
plt.xlabel('Hidden Nodes')
plt.ylabel('Best Accuracy')
plt.plot(nodes, testscore)



#4b
clf = MLPClassifier(hidden_layer_sizes=maxtrainnode, activation='relu', solver='sgd', learning_rate = 'adaptive',alpha= maxtrainalpha, learning_rate_init = maxtrainrate, max_iter=200)
clf.fit(Xtrain, ytrain)
Z = clf.predict(np.c_[x1mesh.ravel(), x2mesh.ravel()])

    # Put the result into a color plot
Z = Z.reshape(x1mesh.shape)

    # Plot the training points with the mesh
plt.figure()
plt.pcolormesh(x1mesh, x2mesh, Z, cmap=cmap_light)
ytrain_colors = [y-1 for y in ytrain]
plt.scatter(Xtrain[:, 0], Xtrain[:, 1], c=ytrain_colors, cmap=cmap_bold, s=20)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title('Neural network')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.figure()
#4d
w11 = np.linspace(-10, 10, 75)
w21 = np.linspace(-10, 10, 75)

# create a meshgrid and evaluate training MSE
W11, W21 = np.meshgrid(w11, w21)
MSEmesh = []
for coef1, coef2 in np.c_[W11.ravel(), W21.ravel()]:
    clf.coefs_[1][0][2] = coef1 
    clf.coefs_[0][0][1] = coef2
    MSEmesh.append( [clf.score(Xtrain,ytrain)] )

MSEmesh = np.array(MSEmesh)

# Put the result into a color plot
MSEmesh = MSEmesh.reshape(W11.shape)

ax = plt.axes(projection='3d')
ax.plot_surface(W11, W21, MSEmesh, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('Training MSE');

#5a

plt.rcParams.update({'figure.max_open_warning': 0})

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap


trainfile = "igits-train.csv"
testfile = "digits-test.csv"

traindata = pd.read_csv(trainfile)
testdata = pd.read_csv(testfile)

ytrain = np.array(traindata.iloc[:,0])
column_numbers = [x for x in range(traindata.shape[1])]
column_numbers.remove(0)
Xtrain = np.array(traindata.iloc[:, column_numbers])


ytest = np.array(testdata.iloc[:,0])
column_numbers = [x for x in range(testdata.shape[1])]
column_numbers.remove(0)
Xtest = np.array(testdata.iloc[:, column_numbers])

scaler = StandardScaler()
scaler.fit(Xtrain)
Xtrain = scaler.transform(Xtrain)
Xtest = scaler.transform(Xtest)

alpharange = np.logspace(-6,0,4)
learnrate = np.logspace(-2,-0.5,4)
trainscore = np.zeros(19)
testscore = np.zeros(19)
nodes = np.zeros(19)
testnode = 0.0
testaccuracy = 0.0
testalpha = 0.0
testrate = 0.0
trainaccuracy = 0.0

maxtestacc = 0.0
maxtrainacc = 0.0

testalpha = 0.0
testrate = 0.0

trainalpha = 0.0
trainrate = 0.0

maxtrainalpha = 0.0
maxtrainrate = 0.0

maxtestalpha = 0.0
maxtestrate = 0.0

maxtrainnode = 0
maxtestnode = 0

noderange = range(5,100,5)

#0,1,2,3, ....18
#5, 10,15, 20?, 25?,
i = 0
 
for value in noderange:
    nodetestacc = 0.0
    nodetrainacc = 0.0
    for a in alpharange:
        for rate in learnrate:
            clf = MLPClassifier(hidden_layer_sizes=value, activation='relu', solver='sgd', learning_rate = 'adaptive',alpha= a, learning_rate_init = rate, max_iter=200)
            clf.fit(Xtrain, ytrain)
            trainacc = clf.score(Xtrain, ytrain)
            testacc = clf.score(Xtest, ytest)
            if(testacc > nodetestacc):
                nodetestacc = testacc
                testscore[i] = testacc
                testalpha = a
                testrate = rate
            if(trainacc > nodetrainacc):
                nodetrainacc = trainacc
                trainscore[i] = trainacc
                trainalpha = a
                trainrate = rate
    nodes[i] = i
    if (maxtestacc < testscore[i]):
        maxtestacc = testscore[i]
        maxtestalpha = testalpha
        maxtestrate = testrate
        maxtestnode = i
        
    if (maxtrainacc < trainscore[i]):
        maxtrainacc = trainscore[i]
        maxtrainalpha = trainalpha
        maxtrainrate = trainrate
        maxtrainnode = i
    i = i + 1
        

print("The best test accuracy  "+ str(maxtestacc) + "with an alpha value pf " + str(maxtestalpha)+ ", a learning rate of " + str(maxtestrate) + ", and " + str(maxtestnode) + " nodes")
# train network with a single hidden layer of 2 nodes
plt.figure()
plt.title('Training Set of the number of hidden nodes')
plt.xlabel('Hidden Nodes')
plt.ylabel('Best Accuracy')
plt.plot(range(5,100,5), trainscore)

plt.figure()
plt.title('Test Set of the number of hidden nodes')
plt.xlabel('Hidden Nodes')
plt.ylabel('Best Accuracy')
plt.plot(range(5,100,5), testscore)

trainscore = np.zeros(50)
testscore = np.zeros(50)
iterations = np.zeros(50)
i = 0
for itr in range(1,51):
    clf = MLPClassifier(hidden_layer_sizes=maxtestnode, activation='relu', solver='sgd', learning_rate = 'adaptive',alpha= 0.0, learning_rate_init = maxtestrate, max_iter=itr)
    clf.partial_fit(Xtrain, ytrain, np.unique(ytrain))

    trainacc = clf.score(Xtrain, ytrain)
    testacc = clf.score(Xtest, ytest)
    
    testscore[i] = testacc
    trainscore[i] = trainacc
    iterations[i] = i
    i = i + 1
        
plt.figure()
plt.title('Training Accuracy of Iterations')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.plot(iterations, trainscore)

plt.figure()
plt.title('Testing Accuracy of Iterations')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.plot(iterations, testscore)

