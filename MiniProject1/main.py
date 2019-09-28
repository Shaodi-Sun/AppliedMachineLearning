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
# print(MalignantCount) 239

breastCancerData = np.array(breastCancerArrayRowsDeleted, dtype=np.float)
breastCancerArrayDropLastColumn = np.delete(breastCancerArrayRowsDeleted, np.s_[-1:], axis=1)
breastCancerFeature = np.delete(np.array(breastCancerArrayDropLastColumn, dtype=np.float), 0, axis = 1)
benignClass = breastCancerFeature[rowsForBenign, :]
malignantClass = breastCancerFeature[rowsForMalignant, :]

classArray = np.zeros(breastCancerArrayRowsDeleted.shape[0])
index = 0 
while index <= classArray.shape[0]: 
    if (index in malignantClass):
        classArray[index] = 1
    index = index + 1


def evaluate_acc(X,y,y_head):
    scsCount = 0
    # print(X[0:3,:])
    # print(y[0:8])
    for i in range(len(y)):
        if (y[i] == y_head[i]):
            scsCount += 1
    print("Accuracy: %.2f %%" % (100 * scsCount / y.shape[0]))
    return 100 * scsCount / y.shape[0]



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
lda = LDA(benignCount, MalignantCount)
start = time.process_time()
test = lda.fit(breastCancerFeature, benignClass, malignantClass, benignCount, MalignantCount)
end = time.process_time()
print("Training Time: %.f s" % (end-start))
n = breastCancerFeature.shape[0]
predictedClass = np.zeros(n)
for i in range(n):
  predictedClass[i] = lda.predict(breastCancerFeature[i, :])
evaluate_acc(breastCancerFeature, classArray, predictedClass)
# Training Time: 0 s
# Accuracy: 98.54 %
print('--------------------')

def KfoldLDA(rawData, rawFeature, Yindex, BV, MV, BC, MC, k):
    ''' 
    k-fold cross validation for LDA, rawData and rawFeature should be arrays;
    Yindex is the position of the dependent variable in the rawData;
    BV stands for benign value and MV stands for malignant value;
    BC stands for benign count and MC stands for malignant count;
    k is the folding number.
    '''
    # for example, rawData => breastCancerData, rawFeature => breastCancerFeature, Yindex => 10, BV = Benign Value => 2, MV = Malignant Value => 4
    dataSize = len(rawData)
    foldSize = int(dataSize / k)

    for i in range(k):
        print('K-fold validation iteration #', i)

        validationFeature = rawFeature[(i * foldSize): ((i + 1) * foldSize)]
        validationData = rawData[(i * foldSize): ((i + 1) * foldSize)]

        trainingFeature  = np.concatenate((rawFeature[: (i * foldSize)], rawFeature[((i + 1) * foldSize):]), axis = 0)
        trainingData  = np.concatenate((rawData[: (i * foldSize)], rawData[((i + 1) * foldSize):]), axis = 0)

        # counting benign/malignant classes in training set
        rowsForBenignKfold = np.where(trainingData[:, Yindex] == BV)[0]
        rowsForMalignantKfold= np.where(trainingData[:, Yindex] == MV)[0]
        benignClassKfold = trainingFeature[rowsForBenignKfold, :]
        malignantClassKfold = trainingFeature[rowsForMalignantKfold, :]

        # validation set for evaluation
        # counting benign/malignant classes in validation set
        rowsForMalignantValidation = np.where(validationData[:, Yindex] == MV)[0]
        malignantClassValidation = validationFeature[rowsForMalignantValidation, :]

        classArrayKfold = np.zeros(validationData.shape[0])
        index = 0 
        while index <= classArrayKfold.shape[0]: 
            if (index in rowsForMalignantValidation):
                classArrayKfold[index] = 1
            index = index + 1

        # Run LDA
        ldaKfold = LDA(BC, MC)
        startKfold = time.process_time()
        testKfold = ldaKfold.fit(trainingFeature, benignClassKfold, malignantClassKfold, BC, MC)
        endKfold = time.process_time()
        print("Training Time: %.f s" % (endKfold-startKfold))
        n = validationFeature.shape[0]
        predictedClassKfold = np.zeros(n)
        for j in range(n):
            predictedClassKfold[j] = ldaKfold.predict(validationFeature[j, :])
        evaluate_acc(validationFeature, classArrayKfold, predictedClassKfold)


KfoldLDA(breastCancerData, breastCancerFeature, 10, 2, 4, benignCount, MalignantCount, 5)
wineData = np.array(wineQualityDataFrameDropLastColumn, dtype=np.float)
wineData = np.c_[wineData, qualityArray]
# test subset of features
wineData = np.delete(wineData, np.s_[5:7], axis = 1)
wineFeatures = np.delete(wineFeatures, np.s_[5:7], axis = 1)
#wineData = np.delete(wineData, np.s_[0:3], axis = 1)
#wineFeatures = np.delete(wineFeatures, np.s_[0:3], axis = 1)
#KfoldLDA(wineData, wineFeatures, 9, 0, 1, wineQualityZero, wineQualityOne, 5)
#print(wineData.shape[1])
#print(wineFeatures.shape[1])