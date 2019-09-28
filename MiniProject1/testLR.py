import pip
from pip._internal import main
import os
#os.system('/bin/bash -c "sudo pip install numpy matplotlib scipy pandas"')
import numpy as np
import math
import scipy
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import time
from classification import logisticRegression
from classification import LDA
from tempfile import TemporaryFile

# main(['list'])
# main(['show', 'wheel'])
# print('scipy Version: '+scipy.__version__)
# print('matplotlib Version: '+matplotlib.__version__)
# print (np.cbrt(27))

#download data file
#wineQualityRawData = os.system('/bin/bash -c "curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"')
#breastCancerRawData = os.system('/bin/bash -c "curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"')

#convert datafile into numpy nd-array
wineQualityDataFrame= pd.read_csv('winequality-red.csv', sep='\t|;|,|[|]', engine='python', header=None).drop(0)
wineQualityDataFrameDropLastColumn = wineQualityDataFrame.iloc[:, :-1]
qualitySeries = wineQualityDataFrame.iloc[:, -1]
quality= pd.to_numeric(qualitySeries, downcast='signed')

#convert into binary task
winePositiveData = wineQualityDataFrameDropLastColumn.loc[quality >= 6]
wineNegativeData = wineQualityDataFrameDropLastColumn.loc[quality < 6]
winePositiveQualityNumpyArray = np.array(winePositiveData,dtype=np.float)
wineNegativeQualityNumpyArray = np.array(wineNegativeData,dtype=np.float)
wineQualityZero = winePositiveQualityNumpyArray.shape[0]
wineQualityOne = wineNegativeQualityNumpyArray.shape[0]


# print(winePisitiveQualityNumpyArray.shape)(855, 11)
# print(wineNegativeQualityNumpyArray.shape)(744, 11)

qualityBinary = (quality>=6).to_numpy()
# print(qualityBinary) (1599,)
wineFeatures = np.array(wineQualityDataFrameDropLastColumn,dtype=np.float)
qualityArray = np.zeros(wineFeatures.shape[0])
index = 0
while index <= qualityArray.shape[0]:
    if (index in wineNegativeQualityNumpyArray):
        qualityArray[index] = 1
    index = index + 1
# print(wineFeatures.shape) (1599, 11)

#print(wineQualityNumpyArray.shape) output (1600, 12)
breastCancerNumpyArray= np.loadtxt('breast-cancer-wisconsin.data', dtype=object, delimiter=',')
# = np.insert(breastCancerNumpyArray, [0], ['Sample code number','Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class'], axis = 0)
#print(breastCancerArrayRowAdded.shape) output(700, 11)
#print(breastCancerArrayRowAdded[24, 6])

#clean data and create binary classification
rowsToDelete = np.where(breastCancerNumpyArray == "?")[0]
breastCancerArrayRowsDeleted = np.delete(breastCancerNumpyArray, rowsToDelete, 0)

# counting benign/malignant classes
benignCount = np.count_nonzero(breastCancerArrayRowsDeleted[:, 10] == "2", axis=0)
MalignantCount = np.count_nonzero(breastCancerArrayRowsDeleted[:, 10] == "4", axis=0)
rowsForBenign = np.where(breastCancerArrayRowsDeleted[:, 10] == "2")[0]
#print('aaaaaaa', rowsForBenign, 'aaaaaaaaa', breastCancerArrayRowsDeleted)
rowsForMalignant= np.where(breastCancerArrayRowsDeleted[:, 10] == "4")[0]
# print(benignCount) 444
# print(MaglignantCount) 239

breastCancerData = np.array(breastCancerArrayRowsDeleted, dtype=np.float)
breastCancerArrayDropLastColumn = np.delete(breastCancerArrayRowsDeleted, np.s_[-1:], axis=1)
breastCancerFeature = np.array(breastCancerArrayDropLastColumn, dtype=np.float)
benignClass = breastCancerFeature[rowsForBenign, :]
malignantClass = breastCancerFeature[rowsForMalignant, :]

classArray = np.zeros(breastCancerArrayRowsDeleted.shape[0])
classArray[rowsForMalignant] = 1

def evaluate_acc(X,y,y_head):
    scsCount = 0
    # print(X[0:3,:])
    # print(y[0:8])
    for i in range(len(y)):
        if (y[i] == y_head[i]):
            scsCount += 1
    print("Accuracy: %.2f %%" % (100 * scsCount / y.shape[0]))
    return 100 * scsCount / y.shape[0]


# kfold validation for logistic regression
def excludeSegment(Xsegments, k):
    '''
    :param Xsegments: a list of segments
    :param k: indicates the kth segment to exclude from the Xsegments list
    :return: a numpy ndarray of all the training data except that from the kth segment
    '''
    trainingSet = np.empty
    for i, segment in enumerate(Xsegments):
        if (i != k):
            if (np.shape(trainingSet)== ()):
                trainingSet = segment
            else:
                trainingSet = np.append(trainingSet, segment, axis=0)
    return trainingSet

# partition
def partition(X,y,k,XSegments,ySegments):
    for i in range(k):
        lower = math.ceil(i * X.shape[0] / k)
        upper = math.ceil((i + 1) * X.shape[0] / k)
        XSegments.append(X[lower:upper, :])
        ySegments.append(y[lower:upper])

def normalize(X):
    X_clone = np.copy(X)
    # normalize features
    for i in range(X.shape[1]):
        X_clone[:,i] = np.divide(X_clone[:,i] - np.min(X_clone[:,i]),np.max(X_clone[:,i]) - np.min(X_clone[:,i]))
    return X_clone

def kfoldLR(X,y,k,a = 0.08, epsilon = 1e-3, power = 1):
    '''
    run kfoldLR(X,y,1) for training on all data and kfold(X,y,k)
    for k fold validation
    '''
    X_normalized = normalize(X)
    if (k == 1):
        lr = logisticRegression(X_normalized.shape[1])
        start = time.process_time()
        nbIterations = lr.fit(X_normalized, y, a, epsilon, power)
        end = time.process_time()
        print("Number of Iterations to converge: %.f" % nbIterations)
        if (nbIterations == np.inf):
            print("Training Time was too long")
        else:
            print("Training Time: %.2f s" % (end-start))
        predictedY = np.zeros(X_normalized.shape[0])
        for i in range(X_normalized.shape[0]):
          predictedY[i] = lr.predict(X_normalized[i,:])
        return np.array([evaluate_acc(X_normalized,y,predictedY),(end-start)])
    else:
        XSegments = []
        ySegments = []
        partition(X_normalized, y, k, XSegments, ySegments)
        # train on k iterations
        # logistic regression
        LR = [logisticRegression(X_normalized.shape[1]) for i in range(k)]
        avgacc = 0
        avgtt = 0
        for i, lr in enumerate(LR):
            Xtraining = excludeSegment(XSegments, i)
            ytraining = excludeSegment(ySegments, i)

            start = time.process_time()
            nbIterations = lr.fit(Xtraining, ytraining, a, epsilon, power)
            end = time.process_time()
            print("Number of Iterations to converge: %.f" % nbIterations)
            if (nbIterations == np.inf):
                print("Training Time was too long")
            else:
                print("Training Time: %.2f s" % (end-start))
            predictedY= np.zeros(XSegments[i].shape[0])
            for j in range(predictedY.shape[0]):
                predictedY[j] = lr.predict(XSegments[i][j, :])
            print(i, 'th k-fold validation')
            avgacc += evaluate_acc(XSegments[i], ySegments[i], predictedY)
            avgtt += end - start
        avgacc /= k
        avgtt /= k
        print("K-fold validation yields average accuracy of %.2f %%" % avgacc)
        print("K-fold validation yields average training time of %.2f s" % avgtt)
        # K-fold validation yields average accuracy of 73.61 %
        return np.array([avgacc,avgtt])

'''
# test initial learning rate
# kfoldLR(wineFeatures,qualityBinary,1)
learningRate = np.arange(0.01,0.3,0.01)
resultWine2 = np.zeros((learningRate.shape[0],2))
for i, a in enumerate(learningRate):
    resultWine2[i,:] = kfoldLR(wineFeatures,qualityBinary,5, a, 1e-3)
outfile = TemporaryFile()
np.save(outfile, resultWine2)
np.save('resultWine',resultWine2)

# # kfoldLR(breastCancerFeature,classArray,1)
resultBC2 = np.zeros((learningRate.shape[0],2))
for i, a in enumerate(learningRate):
    resultBC2[i,:] = kfoldLR(breastCancerFeature,classArray,5, a, 1e-3)
outfile = TemporaryFile()
np.save(outfile, resultBC2)
np.save('resultBC2',resultBC2)


# test decay rate
power = np.arange(1,4,0.5)
resultWinePower = np.zeros((power.shape[0],2))
for i, p in enumerate(power):
    resultWinePower[i,:] = kfoldLR(wineFeatures,qualityBinary,5,0.1,1e-3,p)
np.save('resultWinePower',resultWinePower)

resultBCPower = np.zeros((power.shape[0],2))
for i, p in enumerate(power):
    resultBCPower[i,:] = kfoldLR(breastCancerFeature,classArray,5,0.1,1e-3,p)
np.save('resultBCPower',resultBCPower)

print(classArray)
'''

# test feature
resultWineFeatures = np.zeros((wineFeatures.shape[1],2))
for colToDelete in range(wineFeatures.shape[1]):
    wineFeatures_ = np.delete(wineFeatures, colToDelete, axis = 1)
#     print(wineFeatures.shape)
#     print(wineFeatures_.shape)
    resultWineFeatures = kfoldLR(wineFeatures_,qualityBinary,5, 0.08, 1e-3)
np.save('resultWineFeatures',resultWineFeatures)

# kfoldLR(breastCancerFeature,classArray,5,0.08,1e-3)
