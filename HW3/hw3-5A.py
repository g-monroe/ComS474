import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

# for some sets of input parameters, SVM will be slow to converge.  We will terminate early.
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
#         print(row)
        if len(row)==3:
            ytrain.append( int(row[0]) )
            Xtrain.append( [float(row[1]) , float(row[2]) ])

Xtrain = np.array(Xtrain)
ytrain = np.array(ytrain)
            
#Verify values loaded correctly
# print(ytrain)
# print(Xtrain)


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

for n_neighbors in [1,5,15]:
    # we create an instance of Neighbours Classifier and fit the data.
    
    clf = KNeighborsClassifier(n_neighbors, weights='uniform', algorithm='auto')
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
    plt.title('%i-NN Training Set' % (n_neighbors))
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
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
    
    #Report training and testing accuracies
    print('Working on k=%i'%(n_neighbors))
    trainacc =  clf.score(Xtrain,ytrain) 
    testacc = clf.score(Xtest,ytest) 
    print('\tThe training accuracy is %.2f'%(trainacc))
    print('\tThe testing accuracy is %.2f'%(testacc))

#5-B3 & B4

kval = np.zeros(30)
trainscore = np.zeros(30)
testscore = np.zeros(30)

trainmax = 0
testmax = 0


i = 0
testk = 0
traink = 0
for k in range(1,31):
    neigh = KNeighborsClassifier(n_neighbors=k, weights='uniform', algorithm='auto')
    neigh.fit(Xtrain, ytrain)
    kval[i] = k
    trainscore[i] = neigh.score(Xtrain, ytrain) # reshape does changes the matrix
    testscore[i] = neigh.score(Xtest, ytest)
    if(testmax < testscore[i]):
        testmax = testscore[i]
        trainmax = trainscore[i]
        testk = k
    i = i + 1

plt.figure()
plt.title("Training accuracy as a function of k")
plt.plot(kval, trainscore, color='black')

plt.figure()
plt.title("Testing accuracy as a function of k")
plt.plot(kval, testscore, color='black')


clf = KNeighborsClassifier(testk, weights='uniform', algorithm='auto')
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
plt.title('%i-NN Training Set' % (traink))
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plot the testing points with the mesh
ypred = clf.predict(Xtest)
plt.figure()
plt.pcolormesh(x1mesh, x2mesh, Z, cmap=cmap_light)
ytest_colors = [y-1 for y in ytest]
plt.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest_colors, cmap=cmap_bold, s=30)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title('%i-NN Testing Set' % (testk))
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

#5c1

clf=LinearDiscriminantAnalysis(solver='svd',shrinkage=None,priors=None).fit(Xtrain, ytrain)
clf.score(Xtrain, ytrain)
print(str(clf.score(Xtrain, ytrain)))
print(str(clf.score(Xtest, ytest)))

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
plt.title('Decision regions for training points for LDA')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plot the testing points with the mesh
ypred = clf.predict(Xtest)
plt.figure()
plt.pcolormesh(x1mesh, x2mesh, Z, cmap=cmap_light)
ytest_colors = [y-1 for y in ytest]
plt.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest_colors, cmap=cmap_bold, s=30)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title('Decision regions for testing points for LDA')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')




#5D
clf=QuadraticDiscriminantAnalysis(priors=None,reg_param=0.0).fit(Xtrain, ytrain)
clf.score(Xtrain, ytrain)
print(str(clf.score(Xtrain, ytrain)))
print(str(clf.score(Xtest, ytest)))

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
plt.title('Decision regions for training points for QDA')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plot the testing points with the mesh
ypred = clf.predict(Xtest)
plt.figure()
plt.pcolormesh(x1mesh, x2mesh, Z, cmap=cmap_light)
ytest_colors = [y-1 for y in ytest]
plt.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest_colors, cmap=cmap_bold, s=30)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title('Decision regions for testing points for QDA')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')


#5F




cvals=np.logspace(-4,2,25,base=10)
print(cvals)
for p in [1,2,3,4]:
    cc = np.zeros(25)
    trainscore = np.zeros(25)
    testscore = np.zeros(25)
    
    trainacc = 0
    testacc = 0
    
    
    i = 0
    testc = 0
    trainc = 0
    for c in cvals:
        clf=SVC(C=c,kernel='poly',degree=p,gamma=1.0,coef0=1.0, shrinking=True,probability=False,max_iter=1000)
        clf.fit(Xtrain, ytrain)
        cc[i] = c
        trainscore[i] = clf.score(Xtrain, ytrain) # reshape does changes the matrix
        testscore[i] = clf.score(Xtest, ytest)
        if(testacc < testscore[i]):
            testacc = testscore[i]
            trainacc = trainscore[i]
            testc = c
        i = i + 1
    print("for p = ", p)
    print("Best C value : ", testc)
    print("   Training accuracy = ", trainacc)
    print("   Testing accuracy = ", testacc)
    
    clf=SVC(C=testc,kernel='poly',degree=p,gamma=1.0,coef0=1.0, shrinking=True,probability=False,max_iter=1000)
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
    plt.title('Decision regions for training points for SVC, p = {}'.format(p))
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Plot the testing points with the mesh
    ypred = clf.predict(Xtest)
    plt.figure()
    plt.pcolormesh(x1mesh, x2mesh, Z, cmap=cmap_light)
    ytest_colors = [y-1 for y in ytest]
    plt.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest_colors, cmap=cmap_bold, s=30)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title('Decision regions for testing points for SVC, p = {}'.format(p))
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    #5G
    
    cvals=np.logspace(-4,2,25,base=10)
    
print(cvals)
for p in [1,2,3,4]:
    cc = np.zeros(25)
    trainscore = np.zeros(25)
    testscore = np.zeros(25)
    
    trainacc = 0
    testacc = 0
    
    
    i = 0
    testc = 0
    trainc = 0
    for c in cvals:
        #loop for gamma
        clf=SVC(C=c,kernel='poly',degree=p,gamma=1.0,coef0=1.0, shrinking=True,probability=False,max_iter=1000)
        clf.fit(Xtrain, ytrain)
        cc[i] = c
        trainscore[i] = clf.score(Xtrain, ytrain) # reshape does changes the matrix
        testscore[i] = clf.score(Xtest, ytest)
        if(testacc < testscore[i]):
            testacc = testscore[i]
            trainacc = trainscore[i]
            testc = c
        i = i + 1
    print("for p = ", p)
    print("Best C value : ", testc)
    print("   Training accuracy = ", trainacc)
    print("   Testing accuracy = ", testacc)
    
    clf=SVC(C=testc,kernel='poly',degree=p,gamma=1.0,coef0=1.0, shrinking=True,probability=False,max_iter=1000)
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
    plt.title('Decision regions for training points for SVC, p = {}'.format(p))
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Plot the testing points with the mesh
    ypred = clf.predict(Xtest)
    plt.figure()
    plt.pcolormesh(x1mesh, x2mesh, Z, cmap=cmap_light)
    ytest_colors = [y-1 for y in ytest]
    plt.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest_colors, cmap=cmap_bold, s=30)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title('Decision regions for testing points for SVC, p = {}'.format(p))
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

