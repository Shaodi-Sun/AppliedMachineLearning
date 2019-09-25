import pip
from pip._internal import main
import os
os.system('/bin/bash -c "sudo pip install numpy matplotlib scipy pandas"')
import numpy as np
import math
import scipy
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import time
from classification import logisticRegression 
from classification import LDA

# main(['list'])
# main(['show', 'wheel'])
# print('scipy Version: '+scipy.__version__)
# print('matplotlib Version: '+matplotlib.__version__)
# print (np.cbrt(27))

#download data file
wineQualityRawData = os.system('/bin/bash -c "curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"')
breastCancerRawData = os.system('/bin/bash -c "curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"')

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

benignCount = np.count_nonzero(breastCancerArrayRowsDeleted[:, 10] == "2", axis=0)
MaglignantCount = np.count_nonzero(breastCancerArrayRowsDeleted[:, 10] == "4", axis=0)
breastCancerArrayDropLastColumn= np.delete(breastCancerArrayRowsDeleted, np.s_[-1:], axis=1)
# print(benignCount) 444
# print(MaglignantCount) 239

rowsForBenign = np.where(breastCancerArrayRowsDeleted[:, 10] == "2")[0]
rowsForMaglignant= np.where(breastCancerArrayRowsDeleted[:, 10] == "4")[0]
breastCancerData = np.array(breastCancerArrayDropLastColumn, dtype=np.float)
benignClass = np.array(breastCancerArrayDropLastColumn, dtype=np.float)[rowsForBenign, :]
maglignantClass =  np.array(breastCancerArrayDropLastColumn, dtype=np.float)[rowsForMaglignant, :]

classArray = np.zeros(breastCancerArrayRowsDeleted.shape[0])
index = 0 
while index <= classArray.shape[0]: 
    if (index in maglignantClass):
        classArray[index] = 1
    index = index + 1


def evaluate_acc(X,y,y_head):
    scsCount = 0
    for i in range(len(y)):
        if (y[i] == y_head[i]):
            scsCount += 1
    print("Accuracy: %.2f %%" % (100 * scsCount / n))
    return 100 * scsCount / n

#logistic regression 
#wine dataset
n = wineFeatures.shape[0]
lr = logisticRegression(wineFeatures.shape[1])
start = time.process_time()
w = lr.fit(wineFeatures,qualityBinary,0.05,1e-3)
end = time.process_time()
print("Training Time: %.f s" % (end-start))
predictedQuality = np.zeros(n)
for i in range(n):
  predictedQuality[i] = lr.predict(wineFeatures[i,:])
evaluate_acc(wineFeatures,qualityBinary,predictedQuality)
# With normalization
# Number of Iterations to converge: 225
# Training Time: 5 s
# Accuracy: 74.48 %

#breast cancer dataset
n = breastCancerData.shape[0]
ls = logisticRegression(breastCancerData.shape[1])
start = time.process_time()
w = ls.fit(breastCancerData,classArray,0.05,1e-3)
end = time.process_time()
print("Training Time: %.f s" % (end-start))
perdictedY = np.zeros(n)
for i in range(n):
  perdictedY[i] = ls.predict(breastCancerData[i,:])

evaluate_acc(breastCancerData,classArray,perdictedY)
# Number of Iterations to converge: 168
# Training Time: 1 s
# Accuracy: 98.54 %



#LDA
#wine dataset
lda = LDA(wineQualityZero, wineQualityOne)
start = time.process_time()
test = lda.fit(wineFeatures, winePositiveQualityNumpyArray, wineNegativeQualityNumpyArray, wineQualityZero, wineQualityOne)
end = time.process_time()
print("Training Time: %.f s" % (end-start))
n = wineFeatures.shape[0]
perdictedY = np.zeros(n)
for i in range(n):
  perdictedY[i] = lda.predict(wineFeatures[i,:])
evaluate_acc(wineFeatures,qualityArray,perdictedY)
# Training Time: 0 s
# Accuracy: 91.12 %

#breast cancer dataset
lda = LDA(benignCount, MaglignantCount)
start = time.process_time()
test = lda.fit(breastCancerData, benignClass, maglignantClass, benignCount, MaglignantCount)
end = time.process_time()
print("Training Time: %.f s" % (end-start))
n = breastCancerData.shape[0]
predictedClass = np.zeros(n)
for i in range(n):
  predictedClass[i] = lda.predict(breastCancerData[i,:])
evaluate_acc(breastCancerData,classArray,predictedClass)
# Training Time: 0 s
# Accuracy: 98.54 %

# kfold validation
def excludeSegment(Xsegments, k):
    '''
    :param Xsegments: a list of segments
    :param k: indicates the kth segment to exclude from the Xsegments list
    :return: a numpy ndarray of all the training data except that from the kth segment
    '''
    trainingSet = np.empty
    for i, segment in enumerate(Xsegments):
        # print("size of set of ", i, "is ", segment.shape)
        if (i != k):
            if (np.shape(trainingSet)== ()):
                trainingSet = segment
            else:
                trainingSet = np.append(trainingSet, segment, axis=0)
    # print("size of set for ", k, "is ", trainingSet.shape)
    return trainingSet

k = 5

# partition
# TODO: create function to partition different sets
wineFeaturesKSegments = []
qualityBinarySegments = []
for i in range(k):
    lower = math.ceil(i * wineFeatures.shape[0] / k)
    upper = math.ceil((i + 1) * wineFeatures.shape[0] / k)
    print(lower, ' ', upper )
    print(wineFeatures[lower:upper, :].shape)
    wineFeaturesKSegments.append(wineFeatures[lower:upper, :])
    qualityBinarySegments.append(qualityBinary[lower:upper])
    # TODO: bug - the dimension of the segments are one row fewer than they are supposed to be (319,11) vs (320,11)

# train on k iterations
LR = [logisticRegression(wineFeatures.shape[1]) for i in range(k)]
for i, lr in enumerate(LR):
    X = excludeSegment(wineFeaturesKSegments, i)
    y = excludeSegment(qualityBinarySegments, i)
    lr.fit(X, y, 0.05, 1e-3)
    predictedQuality = np.zeros(wineFeaturesKSegments[i].shape[0])
    for j in range(predictedQuality.shape[0]):
        predictedQuality[j] = lr.predict(wineFeaturesKSegments[i][j, :])
    print(i, 'th k-fold validation')
    evaluate_acc(wineFeaturesKSegments[i], qualityBinarySegments[i], predictedQuality)

