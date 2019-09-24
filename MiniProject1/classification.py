import numpy as np
import math

class logisticRegression:
    """logistic regression model class"""

    def __init__(self,m):
        """ initialize the model parameters as attributues, 
        as well as to define other important properties of the model """
        # initialize model parameters
        
        # w is the m x 1 vector of weights.
        # m: num of features
        self.w = np.random.rand(m)

    def fit(self,X, y, a, epsilon):
        """ Define a fit function, which takes the training data (i.e., X and y)
        —as well as other hyperparameters (e.g., the learning rate and/or number 
        of gradient descent iterations) —as input. 
        This function should train your model by modifying the model parameters """
  
        # X is the n x m matrix of input data, 
        # y is the n x 1 vector of output data,
      
        # n: num of training examples 
        n = X.shape[0]
        # m: num of features
        m = X.shape[1]
       
        # fetch instance model
        w = self.w
        # a: step size or the training rate
        # ak: step size at kth iteration, ak = a0/k
        ak = a
        count = 0 
        deltaW = epsilon + 1

        # gradient descent, minimizing the cross-entropy loss
        while (abs(np.max(deltaW))> epsilon):
            w0 = w
            der = np.zeros((m))
            for i in range(n):
              der += (y[i] - self.logistic(w,X[i,:]))*X[i,:]
            w = w + ak * der
            deltaW = w - w0
            count += 1 
            ak = a/count
#             print(abs(np.max(deltaW)))
        self.w = w
        print("Number of Iterations to converge: %d" % count)
        return w
      
    def predict(self,X):
        """ Define a predict function, which takes a set of input points (i.e., X) 
        as input and outputs predictions (i.e., yˆ) for these points. Note that you
         need to convert probabilities to binary 0-1 predictions by thresholding 
         the output at 0.5! """
        # perdiction results
        log_odds = np.dot(self.w.T,X)
        y_head = (log_odds>0)
        return y_head
      
    def logistic(self,w,Xi):
        """ Compute the logistic function, given w and the features """
        # print(w.T)
        # print(Xi)
        a = np.dot(w.T,Xi)
        return 1/(1+np.exp(-a))

class LDA:
    """LDA model class"""
    
    
    def __init__(self, N0, N1):
        """ initialize the model parameters as attributues,
        as well as to define other important properties of the model """
        # leanrning rate
        #self.w = np.zeros(N);
        self.p0 = N0/(N0+N1) 
        self.p1 = N1/(N0+N1)
        self.mu0 = np.zeros(N0+N1)
        self.mu1 = np.zeros(N0+N1)
        self.covariance = np.zeros([N0+N1, N0+N1])
      
        
    
    def fit(self,X, C1, C2, N0, N1):
        """ Define a fit function, which takes the training data (i.e., X and y)
        —as well as other hyperparameters (e.g., the learning rate and/or number
        of gradient descent iterations)—as input. 
        This function should train your model by modifying the model parameters """
        self.mu0 = np.mean(C1, axis = 0)
        self.mu1 = np.mean(C2, axis = 0)
        denominator = (N0 + N1)/2
        self.covariance = np.zeros([X.shape[1], X.shape[1]])
    
    def predict(self, X):
        """ Define a predict function, which takes a set of input points (i.e., X) 
        as input and outputs predictions (i.e., yˆ) for these points. Note that you
         need to convert probabilities to binary 0-1 predictions by thresholding 
         the output at 0.5! """
        # perdiction results
        covariance_inverse = np.linalg.inv(self.covariance)
        subtract_matrix = np.subtract(self.mu1, self.mu0)
        dot_product = np.dot(np.dot(X.T, covariance_inverse), subtract_matrix)
        w0 = math.log((self.p1)/(self.p0)) - 1/2(np.dot(np.dot(self.mu1.T, covariance_inverse), self.mu1)) + 1/2(np.dot(np.dot(self.mu0.T, covariance_inverse), self.mu0))
        log_odds = w0 + dot_product
        y_head = (log_odds>0)
        return y_head

# help(logisticRegression)
# help(LDA)

