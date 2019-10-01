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
from toolFunctions import evaluate_acc
from toolFunctions import excludeSegment
from toolFunctions import partition
from toolFunctions import normalize
from toolFunctions import pca

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

def kfoldLR(X,y,k,a = 0.08, epsilon = 1e-3, power = 1):
    '''
    :param X: features of dataset, nxm numpy.ndarray
    :param y: class array of dataset, nx1 numpy array
    :param k: k in k-fold cross validations
    :param a: learning rate of logistic regression
    :param epsilon: converging threshold of logistic regression
    :param power: decay rate of the learning rate in logistic regression
    :return: average accuracy and average training time of the k-folc cross validation
    '''

    # run kfoldLR(X,y,1) for training on all data and kfold(X,y,k) for k fold validation

    X_normalized = normalize(X)
    X_normalized = np.c_[X_normalized, np.full(X_normalized.shape[0],1)]
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
        return np.array([evaluate_acc(y,predictedY),(end-start)])
    else:
        XSegments = []
        ySegments = []
        partition(X_normalized, y, k, XSegments, ySegments)
        avgacc = 0
        avgatt = 0
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
            avgacc += evaluate_acc(ySegments[i], predictedY)
            avgtt += end - start
        avgacc /= k
        avgtt /= k
        print("K-fold validation yields average accuracy of %.2f %%" % avgacc)
        print("K-fold validation yields average training time of %.2f s" % avgtt)
        # K-fold validation yields average accuracy of 73.61 %
        return np.array([avgacc,avgtt])
'''
# test initial learning rate
# wine data
learningRate = np.arange(0.01,0.3,0.01)
resultWine = np.zeros((learningRate.shape[0],2))
for i, a in enumerate(learningRate):
    resultWine[i,:] = kfoldLR(wineFeatures,qualityBinary,5, a, 1e-3)
outfile = TemporaryFile()
np.save(outfile, resultWine)
np.save('resultWine',resultWine)

# breast cancer data
resultBC = np.zeros((learningRate.shape[0],2))
for i, a in enumerate(learningRate):
    resultBC[i,:] = kfoldLR(breastCancerFeature,classArray,5, a, 1e-3)
outfile = TemporaryFile()
np.save(outfile, resultBC)
np.save('resultBC',resultBC)

# test decay rate
# wine data
power = np.arange(1,4,0.5)
resultWinePower = np.zeros((power.shape[0],2))
for i, p in enumerate(power):
    resultWinePower[i,:] = kfoldLR(wineFeatures,qualityBinary,5,0.1,1e-3,p)
np.save('resultWinePower',resultWinePower)
# breast cancer data
resultBCPower = np.zeros((power.shape[0],2))
for i, p in enumerate(power):
    resultBCPower[i,:] = kfoldLR(breastCancerFeature,classArray,5,0.1,1e-3,p)
np.save('resultBCPower',resultBCPower)

# test wine data feature: drop one feature
resultWineFeatures = np.zeros((wineFeatures.shape[1],2))
for colToDelete in range(wineFeatures.shape[1]):
    wineFeatures_ = np.delete(wineFeatures, colToDelete, axis = 1)
    resultWineFeatures[colToDelete] = kfoldLR(wineFeatures_,qualityBinary,5, 0.08, 1e-3)
np.save('resultWineFeatures',resultWineFeatures)

# test wine data feature: PCA
kfoldLR(pca(normalize(wineFeatures)),qualityBinary,5,0.08,1e-3)
wineFeatures_ = np.delete(wineFeatures,[5,6], axis = 1)
kfoldLR(wineFeatures_,qualityBinary,5,0.08,1e-3)


# add feature = sqaure of one of the columns
resultWineAddFeature = np.zeros((wineFeatures.shape[1],2))
for colToPower in range(wineFeatures.shape[1]):
    wineFeatures_ = np.c_[wineFeatures, np.power(wineFeatures[:,colToPower],2)]
    resultWineAddFeature[colToPower] = kfoldLR(wineFeatures_,qualityBinary,5, 0.08, 1e-3)
np.save('resultWineAddFeature2',resultWineAddFeature)
'''

print("---------Train on All Data---------")
kfoldLR(wineFeatures, qualityBinary, 1)
kfoldLR(breastCancerFeature, classArray, 1)
print("---------K-fold validation---------")
kfoldLR(wineFeatures,qualityBinary,5,0.08,1e-3)
kfoldLR(breastCancerFeature,classArray,5,0.08,1e-3)
