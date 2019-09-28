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
from toolFunctions import evaluate_acc
from toolFunctions import excludeSegment
from toolFunctions import partition


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
wineQualityDataFrameFeature = wineQualityDataFrame.iloc[:, :-1]
qualitySeries = wineQualityDataFrame.iloc[:, -1]
quality= pd.to_numeric(qualitySeries, downcast='signed')

#convert into binary task
winePositiveData = wineQualityDataFrameFeature.loc[quality >= 6]
wineNegativeData = wineQualityDataFrameFeature.loc[quality < 6]
winePositiveQualityNumpyArray = np.array(winePositiveData,dtype=np.float)
wineNegativeQualityNumpyArray = np.array(wineNegativeData,dtype=np.float)
wineQualityZeroCount = wineNegativeQualityNumpyArray.shape[0]
wineQualityOneCount = winePositiveQualityNumpyArray.shape[0]
#print(wineQualityNumpyArray.shape) output (1600, 12)
# print(winePisitiveQualityNumpyArray.shape)(855, 11)
# print(wineNegativeQualityNumpyArray.shape)(744, 11)

qualityBinary = (quality>=6).to_numpy()
wineFeatures = np.array(wineQualityDataFrameFeature,dtype=np.float)
# print(qualityBinary) (1599,)
# print(wineFeatures.shape) (1599, 11)

# load breast cancer data
breastCancerNumpyArray= np.loadtxt('breast-cancer-wisconsin.data', dtype=object, delimiter=',')
# = np.insert(breastCancerNumpyArray, [0], ['Sample code number','Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class'], axis = 0)
#print(breastCancerArrayRowAdded.shape) output(700, 11)
#print(breastCancerArrayRowAdded[24, 6])

# clean data and create binary classification
rowsToDelete = np.where(breastCancerNumpyArray == "?")[0]
breastCancerArrayCleaned = np.delete(breastCancerNumpyArray, rowsToDelete, 0)

# counting benign/malignant classes
benignCount = np.count_nonzero(breastCancerArrayCleaned[:, 10] == "2", axis=0)
MalignantCount = np.count_nonzero(breastCancerArrayCleaned[:, 10] == "4", axis=0)
rowsForBenign = np.where(breastCancerArrayCleaned[:, 10] == "2")[0]
rowsForMalignant= np.where(breastCancerArrayCleaned[:, 10] == "4")[0]
# print(benignCount) 444
# print(MalignantCount) 239

breastCancerData = np.array(breastCancerArrayCleaned, dtype=np.float)
breastCancerArrayFeature = np.delete(breastCancerArrayCleaned, np.s_[-1:], axis=1)
breastCancerFeature = np.delete(np.array(breastCancerArrayFeature, dtype=np.float), 0, axis = 1)
benignClass = breastCancerFeature[rowsForBenign, :]
malignantClass = breastCancerFeature[rowsForMalignant, :]

classArray = np.zeros(breastCancerArrayCleaned.shape[0])
classArray[rowsForMalignant] = 1

def KfoldLDA(X, y, k):
    ''' 
    :param X: features of dataset, nxm numpy.ndarray
    :param y: classe array of dataset, nx1 numpy array
    :param k: k in k-fold cross validations
    :return: average accuracy and average training time of the k-folc cross validation
    '''
    if (k == 1):
        rowsForPositive = np.where(y[:] == 1)[0]
        rowsForNegative = np.where(y[:] == 0)[0]
        positiveClass = X[rowsForPositive, :]
        negativeClass = X[rowsForNegative, :]
        PC = rowsForPositive.shape[0]
        NC = rowsForNegative.shape[0]

        # Run LDA
        lda = LDA(NC, PC)
        start = time.process_time()
        lda.fit(X, negativeClass, positiveClass, NC, PC)
        end = time.process_time()
        print("Training Time: %.f s" % (end - start))
        n = X.shape[0]
        predictedClass = np.zeros(n)
        for i in range(n):
            predictedClass[i] = lda.predict(X[i, :])
        return np.array([evaluate_acc(y, predictedClass),end-start])
    else:
        XSegments = []
        ySegments = []
        partition(X, y, k, XSegments, ySegments)
        avgacc = 0
        avgtt = 0
        for i in range(k):
            print('K-fold validation iteration #', i)
            trainingFeatures = excludeSegment(XSegments, i)
            trainingClass = excludeSegment(ySegments, i)
            validationFeatures = XSegments[i]
            validationClass = ySegments[i]

            # counting benign/malignant classes in training set
            rowsForPositive = np.where(trainingClass[:] == 1)[0]
            rowsForNegative = np.where(trainingClass[:] == 0)[0]
            positiveClass = trainingFeatures[rowsForPositive, :]
            negativeClass = trainingFeatures[rowsForNegative, :]
            PC = rowsForPositive.shape[0]
            NC = rowsForNegative.shape[0]

            # Run LDA
            lda = LDA(NC, PC)
            start = time.process_time()
            lda.fit(trainingFeatures,negativeClass,positiveClass, NC, PC)
            end = time.process_time()
            print("Training Time: %.f s" % (end-start))
            n = validationFeatures.shape[0]
            predictedClass = np.zeros(n)
            for j in range(n):
                predictedClass[j] = lda.predict(validationFeatures[j, :])
            avgacc += evaluate_acc(validationClass, predictedClass)
            avgtt += end - start
            del lda
        avgacc /= k
        avgtt /= k
        print("K-fold validation yields average accuracy of %.2f %%" % avgacc)
        print("K-fold validation yields average training time of %.2f s" % avgtt)
        return np.array([avgacc, avgtt])

KfoldLDA(breastCancerFeature, classArray, 1)
KfoldLDA(wineFeatures, qualityBinary, 1)
print("---------")
KfoldLDA(breastCancerFeature, classArray, 5)
KfoldLDA(wineFeatures, qualityBinary, 5)
