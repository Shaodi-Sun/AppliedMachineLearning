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
winePositiveData = wineQualityDataFrameDropLastColumn.loc[quality >= 6]
wineNegativeData = wineQualityDataFrameDropLastColumn.loc[quality < 6]

winePisitiveQualityNumpyArray = winePositiveData.to_numpy()
wineNegativeQualityNumpyArray = wineNegativeData.to_numpy()

# print(winePisitiveQualityNumpyArray.shape)(855, 11)
# print(wineNegativeQualityNumpyArray.shape)(744, 11)

#print(wineQualityNumpyArray.shape) output (1600, 12)
breastCancerNumpyArray= np.loadtxt('breast-cancer-wisconsin.data', dtype=object, delimiter=',')
breastCancerArrayRowAdded = np.insert(breastCancerNumpyArray, [0], ['Sample code number','Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class'], axis = 0)
#print(breastCancerArrayRowAdded.shape) output(700, 11)
#print(breastCancerArrayRowAdded[24, 6])

#clean data and create binary classification
rowsToDelete = np.where(breastCancerArrayRowAdded == "?")[0]
breastCancerArrayRowsDeleted = np.delete(breastCancerArrayRowAdded, rowsToDelete, 0)

