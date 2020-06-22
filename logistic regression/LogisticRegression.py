import numpy as np
import random


class LogisticRegression(object):
    
    def __init__(self, alpha=0.01, numIter=10000, treshold=0.5, fitIntercept=True, verbose=False):
        self.alpha = alpha
        self.numIter = numIter
        self.treshold = treshold
        self.fitIntercept = fitIntercept
        self.verbose = verbose
        self.weights = None
        self.X_ = None
        self.y_ = None        
        self.predicted_ = None        
    
    def _addIntercept(self, X):
        intercept = np.ones((X.shape[0],1))   ## for giving intercept
        concat = np.hstack((intercept , X))   ## equal to np.concatenate((intercept, X), axis=1)
        return concat
    
    def _sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    def _loss(self, y, h):
        """log loss function where h is the predicted value """
        if (h==0).any() or (h==1).any(): ## problem of log(0)
            h[h==0] = 1e-10
            h[h==1] = 1e-10        
        return (-y * np.log(h) - (1 - y) * np.log(1 - h))/len(h)
    
    def logLikelihood(self):
        """maximum log likelihood estimation"""
        y = np.mat(self.y_)
        X = np.mat(self.X_)
        Beta = self.weights        
        ll = -1*(np.ones(len(self.y_))*np.log(1 + np.exp(X * Beta)) - y*X*Beta)
        return np.ravel(ll)[0]  
    
    def _newtonsMethod(self,X, y):
        """newton's method optimization function"""
        Xmat = np.mat(X, dtype=np.float64)
        ymat = np.mat(y, dtype=np.float64)
        m, n = Xmat.shape
        weights = np.mat(np.zeros((n , 1)), dtype=np.float64) 
        ### gradient matrix
        def _nablaF(beta):
            return Xmat.T * (self._sigmoid(Xmat*beta) - ymat.T)
        ###  hessian matrix 
        def _nabla2F(beta):
            ## print(np.isfinite(mat).all())  problem of infinity
            with np.errstate(divide="ignore"):
                mat = np.power(1+np.exp(Xmat*beta),2)            
            if (mat == np.inf).any():
                mat[mat==np.inf] = 1e20
            return Xmat.T*(np.diag(np.ravel(np.exp(Xmat*beta)/mat))*Xmat) 
                    
        for i in range(self.numIter):
            weights = weights - np.linalg.inv(_nabla2F(weights))* _nablaF(weights)
            h = self._sigmoid(Xmat * weights)
            if (self.verbose and i% 1000 == 0):
                print('loss: %.4f'% (self._loss(ymat, h)))
        return weights        
    
    def _gradAscent(self, X, y):     
        """gradient ascent optimization functions"""
        Xmat = np.mat(X)
        ymat = np.mat(y).T
        m, n = Xmat.shape
        weights = np.ones((n , 1))
        for i in range(self.numIter):
            h = self._sigmoid(Xmat * weights) ## equal to np.dot(dataMatrix,weights)
            error = (ymat - h)
            weights = weights + self.alpha * Xmat.T * error
            if (self.verbose and i% 1000 == 0) :                
                print('loss: %.4f'% (self._loss(ymat.T,h)))
        return weights
       
    def _stocGradAscent(self, X, y):
        """stochastic gradient ascent"""
        m, n = X.shape
        weights = np.ones(n)
        for j in range(self.numIter):
            H = np.zeros((m,1))
            for i in range(m):
                h = self._sigmoid(sum(X[i]*weights))
                H[i,0] = h
                error = y[i] - h
                weights = weights + self.alpha * error * X[i]
            if (self.verbose and j% 1000 == 0) :                
                print('loss: %.4f'% (self._loss(np.mat(y), H)))
        return np.mat(weights).T
    
    def _modStocGradAscent(self, X, y):
        """modified stochastic gradient ascent"""
        m, n = X.shape
        weights = np.ones(n)
        for j in range(self.numIter): 
            dataIndex = list(range(m))
            for i in range(m):
                modAlpha = 4/(1.0+j+i) + self.alpha              # alpha changes on each iteration
                randIndex = random.choice(dataIndex)        # randomly select one row
                h = self._sigmoid(sum(X[randIndex]*weights))
                error = y[randIndex] - h
                weights = weights + modAlpha * error * X[randIndex]
                dataIndex.remove(randIndex)                 # remove the selected row   
        return np.mat(weights).T
    
    def fit(self, X, y, optimizer = "modified"):        
        if self.fitIntercept:
            X = self._addIntercept(X)    
        self.X_ = X
        self.y_ = y
        optimizers = ["modified", "stochastic", "normal", "newton"]
        functions = [self._modStocGradAscent, self._stocGradAscent, self._gradAscent, self._newtonsMethod]
        if optimizer in optimizers:
            self.weights = functions[optimizers.index(optimizer)](self.X_, self.y_) 
            return("LogisticRegression(optimizer=%s, alpha=%.4f, numIter=%d, treshold=%.2f, fitIntercept=%r)"%(optimizer,self.alpha,self.numIter,self.treshold,self.fitIntercept))
        else:
            raise Exception("Unknown gradient ascent optimization function. Valid names are: modified, stochastic, normal and newton.")
    
    def _probToClass(self, x):
        if (x > self.treshold):  return 1
        else: return 0
        
    def score(self, X = None): 
        """provides the accuracy of trained model"""
        if X is not None:
            X = self._addIntercept(X)          
            self.X_ = X            
        probMat = self._sigmoid(self.X_ * self.weights)        
        classMat = np.vectorize(self._probToClass)(probMat) ## apply a function to each element in a 2d numpy array/matrix
        prediction = np.ravel(classMat)
        self.predicted_ = prediction
        accuracy = [pred==true for pred,true in zip(prediction, self.y_)].count(True)/len(self.y_)
        return accuracy
    
    def predict_prob(self, X):
        return np.ravel(self._sigmoid(self._addIntercept(X) * self.weights)) # returns contiguous flattened 1D array
    
    def prediction(self, X):
        self.score(X)
        return self.predicted_