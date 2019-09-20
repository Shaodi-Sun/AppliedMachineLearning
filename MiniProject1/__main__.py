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

class logisticRegression:
    """logistic regression model class"""
    

    def __init__(self, N):
        """ initialize the model parameters as attributues, as well as to define other important properties of the model """
        # leanrning rate
        self.w = np.zeros(N);

    def fit(X, y, a):
        """ Define a fit function, which takes the training data (i.e., X and y)—as well as other hyperparameters (e.g., the learning rate and/or number of gradient descent iterations)—as input. This function should train your model by modifying the model parameters """

    def perdict(X):
        """ Define a predict function, which takes a set of input points (i.e., X) as input and outputs predictions (i.e., yˆ) for these points. Note that you need to convert probabilities to binary 0-1 predictions by thresholding the output at 0.5! """
        # perdiction results
        return y_head

class LDA:
    """LDA model class"""
    
    
    def __init__(self, N):
        """ initialize the model parameters as attributues, as well as to define other important properties of the model """
        # leanrning rate
        self.w = np.zeros(N);
    
    def fit(X, y, a):
        """ Define a fit function, which takes the training data (i.e., X and y)—as well as other hyperparameters (e.g., the learning rate and/or number of gradient descent iterations)—as input. This function should train your model by modifying the model parameters """
    
    def perdict(X):
        """ Define a predict function, which takes a set of input points (i.e., X) as input and outputs predictions (i.e., yˆ) for these points. Note that you need to convert probabilities to binary 0-1 predictions by thresholding the output at 0.5! """
        # perdiction results
        return y_head

help(logisticRegression)
help(LDA)
