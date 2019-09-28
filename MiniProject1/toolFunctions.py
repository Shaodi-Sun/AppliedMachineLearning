import math
import numpy as np

def evaluate_acc(y, y_head):
    '''
    :param y: true class
    :param y_head: perdicted class
    :return: accuracy of perdiction in percentage
    '''
    scsCount = 0
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
            if (np.shape(trainingSet) == ()):
                trainingSet = segment
            else:
                trainingSet = np.append(trainingSet, segment, axis=0)
    return trainingSet

# partition
def partition(X, y, k, XSegments, ySegments):
    '''
    :param X: features of dataset, nxm numpy.ndarray
    :param y: class array of dataset, nx1 numpy array
    :param k: k in k-fold cross validations
    :param XSegments: list of np.ndarray, containing k segments of X
    :param ySegments: list of np.array, containing k segments of y
    :return:
    '''
    for i in range(k):
        lower = math.ceil(i * X.shape[0] / k)
        upper = math.ceil((i + 1) * X.shape[0] / k)
        XSegments.append(X[lower:upper, :])
        ySegments.append(y[lower:upper])