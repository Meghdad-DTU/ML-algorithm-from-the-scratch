import numpy as np 

class SVM_linear(object):
    def __init__(self):
        self.X = None
        self.W = None
        self.predicted = None
                    
    def _addIntercept(self, X):
        intercept = np.ones((X.shape[0],1))   ## for giving intercept
        concat = np.hstack((intercept , X))   ## equal to np.concatenate((intercept, X), axis=1)
        return concat

    def _hingeLoss(self, X, y, alpha, maxIter):
        """hinge loss function"""
        X = self._addIntercept(X)
        m, n  = X.shape
        W = np.zeros((1,n))
        epochs = 1             ## regularization parameter λ is set to 1/epochs
        while(epochs < maxIter):
            labDist = X@W.T    ## y*|wTx + b|/||w|| : we ignore the vector norm ||w||
            prod = np.multiply(labDist,y)
            count = 0
            for val in prod:
                if(val >= 1):  ## Gradient Update — No misclassification
                    cost = 0
                    W = W - alpha * (2 * 1/epochs * W)                       
                else:          ## Gradient Update — Misclassification
                    cost = 1 - val 
                    W = W + alpha * (y[count]*X[count] - 2 * 1/epochs * W)           
                count += 1
            epochs += 1
        return W
        
    def fit(self, X, y, alpha=0.0001, maxIter = 5000):
        y = np.ravel(y)
        assert list(set(y)) == [0,1], "target variable must be 0 and 1"
        y = np.array([1 if val == 1 else -1 for val in y]).reshape(-1,1)
        self.X = X        
        self.y = y
        W = self._hingeLoss(X, y, alpha, maxIter)
        self.W = W
        return  "SVM(kernel= linear, alpha=%.4f, maxIter=%d)"%(alpha,maxIter) 
    
    def score(self, X= None):
        if X is not None:
            self.X = X
        pred = self._addIntercept(self.X)@self.W.T
        yPred = np.array([1 if val >= 0 else -1 for val in pred])
        self.predicted = np.array([1 if val == 1 else 0 for val in yPred]) ## change from -1,1 to 0,1
        return [pred==true for pred,true in zip(yPred, self.y)].count(True)/len(self.y)
    
    def prediction(self, X):
        self.score(X)
        return self.predicted     