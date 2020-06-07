#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 21:55:11 2020

@author: gavinmonroe
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
plt.rcParams.update({'figure.max_open_warning': 0})
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# for some sets of input parameters, SVM will be slow to converge.  We will terminate early.
# This code will suppress warnings.
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", ConvergenceWarning)

import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
plt.rcParams.update({'figure.max_open_warning': 0})

# for some sets of input parameters, SVM will be slow to converge.  We will terminate early.
# This code will suppress warnings.
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", ConvergenceWarning)
# load training set
# load training set

ytrain = []
Xtrain = []
with open('HW3train.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',',lineterminator='\n')
    for row in reader:
#         print(row)
        if len(row)==3:
            ytrain.append( int(row[0]) )
            Xtrain.append( [float(row[1]) , float(row[2]) ])

Xtrain = np.array(Xtrain)
ytrain = np.array(ytrain)
            
#Verify values loaded correctly
# print(ytrain)
# print(Xtrain)

"""
for x,y in zip(Xtrain, ytrain):
#     print(x1,x2,y)
    if y==1:
        col = 'blue'
    if y==2:
        col = 'red'
    if y==3:
        col = 'black'
    plt.scatter(x[0], x[1],  color=col)
    
plt.title('Scatterplot HW3Train')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
"""
# plt.savefig('HW3trainscatter.png',dpi=300,bbox_inches='tight')

# load testing set
ytest = []
Xtest = []
with open('HW3test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',',lineterminator='\n')
    for row in reader:
#         print(row)
        if len(row)==3:
            ytest.append( int(row[0]) )
            Xtest.append( [float(row[1]) , float(row[2]) ])
            
Xtest = np.array(Xtest)
ytest = np.array(ytest)
            
#Verify values loaded correctly
# print(ytest)
# print(Xtest)
"""
for x,y in zip(Xtest, ytest):
#     print(x1,x2,y)
    if y==1:
        col = 'blue'
    if y==2:
        col = 'red'
    if y==3:
        col = 'black'
    plt.scatter(x[0], x[1],  color=col)
    
plt.title('Scatterplot HW3Test')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
# plt.savefig('HW3trainscatter.png',dpi=300,bbox_inches='tight')
"""



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

for n_neighbors in [1,2,3,4,10]:
    # we create an instance of Neighbours Classifier and fit the data.
    
    clf=DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=n_neighbors)
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
    plt.title('%i-Decision Tree' % (n_neighbors))
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    """
    # Plot the testing points with the mesh
    ypred = clf.predict(Xtest)
    plt.figure()
    plt.pcolormesh(x1mesh, x2mesh, Z, cmap=cmap_light)
    ytest_colors = [y-1 for y in ytest]
    plt.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest_colors, cmap=cmap_bold, s=30)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title('%i-NN Testing Set' % (n_neighbors))
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    """
    #Report training and testing accuracies
   # print('   Working on k=%i'%(n_neighbors))
   # trainacc =  clf.score(Xtrain,ytrain) 
   # testacc = clf.score(Xtest,ytest) 
   # print('\tThe training accuracy is %.2f'%(trainacc))
    #print('\tThe testing accuracy is %.2f'%(testacc))
    
    #4b
    #trying to find accuracy at each depth
acc = np.zeros(10)
acc_test= np.zeros(10)
depthval = np.zeros(10)
maxtrain_acc = 0.0
traindepth = 0
test_max_acc = 0
testdepth = 0


i = 0

for k in range(1,11):
    clf=DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=k)
    clf.fit(Xtrain, ytrain)
    acc[i] = clf.score(Xtrain, ytrain)
    acc_test[i] = clf.score(Xtest, ytest)
    if acc[i] > maxtrain_acc:
        maxtrain_acc = acc[i]
        traindepth = k
    if acc_test[i] > test_max_acc:
        test_max_acc = acc_test[i]
        testdepth = k
    depthval[i] = k
    i+=1

plt.figure()
plt.title("Training accuracy for tree depth")
plt.plot(depthval, acc, color='black')

plt.figure()
plt.title("Testing accuracy for tree depth")
plt.plot(depthval, acc_test, color='black')

print("The training tree depth is " + str(traindepth) + " which has an accuracy of " + str(maxtrain_acc))
print("The test tree depth is " + str(testdepth) + " which has an accuracy of " + str(test_max_acc))


#4c
bestvalue = 0.0
besttree = 0.0
best_number_trees = np.zeros(5)


i = 0

for value in [1,2,3,4,10]:
    best_train_acc = 0.0
    for num_trees in [1,5,10,25, 50, 100, 200]:
      clf=RandomForestClassifier(bootstrap=True,n_estimators=num_trees,max_features=None,criterion='gini',max_depth=value)
      clf.fit(Xtrain, ytrain)
      if clf.score(Xtrain, ytrain) > best_train_acc:
        best_number_trees[i] = num_trees
        best_train_acc = clf.score(Xtrain, ytrain)
    best_tree_number = (int)(best_number_trees[i])
    print(best_tree_number)
  
    clf=RandomForestClassifier(bootstrap=True,n_estimators=best_tree_number,max_features=None,criterion='gini',max_depth=value)
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
    plt.title("Decision tree for the training set at depth " + str(value))
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    i+=1    
    
    
    #4d
#get the accuracies at each level and each depth. Each number of trees. 
    #need a double a for loop 

acc = np.zeros(10)
acc_test= np.zeros(10)
depthval = np.zeros(10)
maxtrain_acc = 0.0
traindepth = 0
test_max_acc = 0
testdepth = 0


i = 0

for k in range(1,11):
    for value in [1,5,10,25,50,100,200]:
        clf=RandomForestClassifier(bootstrap=True,n_estimators=value,max_features=None,criterion='gini',max_depth=k)
        clf.fit(Xtrain, ytrain)
        if acc[i] < clf.score(Xtrain, ytrain):
              maxtrain_acc = acc[i]
              traindepth = k
        if acc[i] < clf.score(Xtrain, ytrain):
            acc[i] = clf.score(Xtrain, ytrain)
            acc_test[i] = clf.score(Xtest, ytest)
        if acc[i] > maxtrain_acc:
            maxtrain_acc = acc[i]
            traindepth = k
        if acc_test[i] > test_max_acc:
            test_max_acc = acc_test[i]
            testdepth = k
            depthval[i] = k
       
    i+=1

plt.figure()
plt.title("Training accuracy for tree depth")
plt.plot(depthval, acc, color='black')

plt.figure()
plt.title("Testing accuracy for tree depth")
plt.plot(depthval, acc_test, color='black')

print("The training tree depth is " + str(traindepth) + " which has an accuracy of " + str(maxtrain_acc))
print("The test tree depth is " + str(testdepth) + " which has an accuracy of " + str(test_max_acc))

#4E
bestvalue = 0.0
besttree = 0.0
best_number_trees = np.zeros(5)


i = 0

for value in [1,2,3,4,10]:
    best_train_acc = 0.0
    for num_trees in [1,5,10,25, 50, 100, 200]:
      rate=np.logspace(-3,0,15,base=10)
      clf=GradientBoostingClassifier(learning_rate=rate,n_estimators=num_trees,max_depth=value)
      clf.fit(Xtrain, ytrain)
      
      if acc[i] < clf.score(Xtrain, ytrain):
             maxtrain_acc = acc[i]
             traindepth = value
      if clf.score(Xtrain, ytrain) > best_train_acc:
        best_number_trees[i] = num_trees
        best_train_acc = clf.score(Xtrain, ytrain)
      best_tree_number = (int)(best_number_trees[i])
      print(best_tree_number)
      
    clf=RandomForestClassifier(bootstrap=True,n_estimators=best_tree_number,max_features=None,criterion='gini',max_depth=traindepth)
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
    plt.title("Boosted tree for the training set at depth " + str(value))
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    i+=1  
  