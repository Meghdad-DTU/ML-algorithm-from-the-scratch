import numpy as np

class OLS_Regression (object):    
    def __init__(self, fit_intercept=True, normalize = False, method="OLS", eps=0.01, alpha=1, maxIter=1000):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.method = method
        self.eps = eps         # epsilon for gradient method
        self.alpha = alpha     # learning rate for gradient method
        self.maxIter = maxIter # Iteration for gradient method  
        self.intercept_ = None
        self.coef_ = None 
        self.yMean = 0
        self.yStd = 1
   
    def _addIntercept(self, X):
        intercept = np.ones((X.shape[0],1))   ## for giving intercept
        concat = np.hstack((intercept , X))   ## equal to np.concatenate((intercept, X), axis=1)
        return concat
    
    def _normalize(self, X):
        xMean = np.mean(X, axis=0)
        xStd = np.std(X, axis=0)
        return (X-xMean)/xStd
    
    def _gradient (self, X, y):
        """Gradient descent algorithm"""
        N, M = X.shape
        ws = np.zeros((M,1)) # Initialize weights
        for i in range(self.maxIter):
            hx = X@ws        
            error = y - hx 
            olsLoss = np.sqrt(sum(np.ravel(error)**2))
            if (i%100==0):
                print("iteration: %3d & loss: %.3f"%(i,olsLoss))
            gradient = (-2/N)*X.T@error         
            ws = ws - self.alpha*gradient 
            if np.sqrt(sum(np.ravel(gradient)**2)) <= self.eps:
                break
        return ws   
   
    def _OLS (self, X, y):
        """Ordinary least square method"""
        xTx = X.T@X
        if np.linalg.det(xTx) == 0.0:
            print ("This matrix is singular, cannot do inverse")
            return
        ws = np.linalg.inv(xTx)@(X.T@y)        
        return ws        
    
    def fit(self, xArr, yArr):
        if self.normalize is True:
            xArr = self._normalize(xArr)
            self.yMean = yArr.mean()
            self.yStd = yArr.std()
            yArr = self._normalize(yArr)            
        
        if  self.fit_intercept is True:            
            xArr = self._addIntercept(xArr)
            
        xMat = np.mat(xArr)
        yMat = np.mat(yArr).T
        methodName = ["OLS", "gradient"]
        method = [self._OLS, self._gradient] 
        if self.method in methodName:
            ind = methodName.index(self.method)
            self.ws = method[ind](xMat, yMat)
        else:
            raise Exception("Unknown method %s. Valid names are: OLS & gradient."%self.method)        
        
        if self.fit_intercept is True:
            self.intercept_ = np.ravel(self.ws[0])[0]
            self.coef_ = np.ravel(self.ws)[1:]
        else:
            self.intercept_ = 0
            self.coef_ = np.ravel(self.ws)    
        return ("OLS_Regression(fit_intercept=%r, method=%s)"%(self.fit_intercept, self.method))
        
    def prediction(self, xArr):
        if self.normalize is True:
            xArr = self._normalize(xArr)
            
        if  self.fit_intercept is True: 
            xArr = self._addIntercept(xArr)
            
        xMat = np.mat(xArr)
        return (np.ravel(xMat@self.ws)*self.yStd + self.yMean)
        
        
class LWLR_Regression(object):
    """Locally weight linear regression:
    it gives a weight to data points near our data point of interest"""
    
    def __init__(self, kernel="guassian", k=1):
        self.kernel = kernel
        self.k = k
    
    def _gaussianKernel(self, x1, x2, sigma=1):
        '''returns the dot product in infinite dimensional space'''
        norm = np.linalg.norm(np.subtract(x1,x2), axis = 1) #norm 
        res = np.exp(-(norm**2)/(2*(sigma**2)))   #returning the final dot product.
        return res
    
    def _polynomialKernel(self, x1, x2, degree=1):
        '''returns the dot in trnasformed polynomial space'''
        dotProduct = np.dot(x1,x2.T)             #give the dot product
        return np.power((1+dotProduct),degree)
    
    def _LWLR(self, testPoint, xArr, yArr):
        # testPoint: only one point 
        xMat = np.mat(xArr)
        yMat = np.mat(yArr).T
        m = np.shape(xMat)[0]
        weights = np.eye(m)
        kernel = [self._gaussianKernel, self._polynomialKernel]
        if self.kernel == "guassian":
            w = kernel[0](x1= np.tile(testPoint,(m,1)), x2= xMat, sigma= self.k)
        elif self.kernel == "polynomial":
            w = kernel[1](x1= np.tile(testPoint,(m,1)), x2= xMat, degree= self.k)
        else:
             raise Exception("Unknown kernel %s. Valid names are: guassian & polynomial."%self.kernel)
        
        np.fill_diagonal(weights, w) # Replace the diagonal element by w
        xTx = xMat.T@(weights@xMat)
        if np.linalg.det(xTx) == 0.0:
            print ("This matrix is singular, cannot do inverse")
            return
        ws = np.linalg.inv(xTx)@(xMat.T@(weights@yMat))
        return testPoint@ws ## predicted value for the test point
    
    def prediction(self, xTest, xTrain, yTrain):
        m = np.shape(xTest)[0]
        yHat = np.zeros(m)
        for i in range(m):
            yHat[i] = self._LWLR(xTest[i], xTrain, yTrain)
        return yHat
        
class Ridge_Regression(object):
    def __init__(self, fit_intercept=True, normalize = True, method = "OLS", lamb =1.0, eps= 0.01, alpha= 0.01, maxIter= 1000):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.method = method        
        self.lamb = lamb
        self.eps = eps
        self.alpha = alpha     # learning rate for gradient method
        self.maxIter = maxIter # iteration for gradient method        
        self.coef_ = None
        self.intercept_ = None        
        self.yMean = 0
        self.yStd = 1        
        
    def _addIntercept(self, X):
        intercept = np.ones((X.shape[0],1))   ## for giving intercept
        concat = np.hstack((intercept , X))   ## equal to np.concatenate((intercept, X), axis=1)
        return concat
    
    def _normalize(self, X):
        xMean = np.mean(X, axis=0)
        xStd = np.std(X, axis=0)
        return (X-xMean)/xStd
    
    def _OLS(self, X, y):  
        """Ordinary least square method"""
        xTx = X.T@X
        denom = xTx + np.eye(np.shape(X)[1])*self.lamb
        if np.linalg.det(denom) == 0.0:
            print ("This matrix is singular, cannot do inverse")
            return
        ws = np.linalg.inv(denom)@(X.T@y)         
        return ws 
    
    def _gradient(self, X, y):
        """Gradient descent algorithm"""
        N, M = X.shape
        ws = np.zeros((M,1)) # Initialize weights
        for i in range(self.maxIter):
            hx = X@ws        
            error = y - hx 
            olsLoss = np.sqrt(sum(np.ravel(error)**2))
            if (i%100==0):
                print("iteration: %3d & loss: %.3f"%(i,olsLoss))
            gradient = (-2/N)*(X.T@error +  np.eye(M)@ws*self.lamb)       
            ws = ws - self.alpha*gradient 
            if np.sqrt(sum(np.ravel(gradient)**2)) <= self.eps:
                break
        return ws
        
    def fit(self, xArr, yArr):                
        if self.normalize is True:
            xArr = self._normalize(xArr)
            self.yMean = yArr.mean()
            self.yStd = yArr.std()
            yArr = self._normalize(yArr)            
        
        if  self.fit_intercept is True:            
            xArr = self._addIntercept(xArr)           
            
        xMat = np.mat(xArr)        
        yMat = np.mat(yArr).T
        
        methodName = ["OLS","gradient"]
        method = [self._OLS,self._gradient] 
        if self.method in methodName:
            ind = methodName.index(self.method)
            self.ws = method[ind](xMat, yMat)
        else:
            raise Exception("Unknown method %s. Valid names are: OLS & gradient."%self.method)        
        
        if self.fit_intercept is True:
            self.intercept_ = np.ravel(self.ws[0])[0]
            self.coef_ = np.ravel(self.ws)[1:]
        else:
            self.intercept_ = 0
            self.coef_ = np.ravel(self.ws)            
        
        return ("Ridge_Regression(fit_intercept=%r, normalize = %r, method=%s, lambda=%.2f)"%(self.fit_intercept,self.normalize,self.method,self.lamb))
    
    def prediction(self, xArr):        
        if self.normalize is True:
            xArr = self._normalize(xArr)
            
        if self.fit_intercept is True:             
            xArr = self._addIntercept(xArr)            
       
        xMat = np.mat(xArr)
        return (np.ravel(xMat@self.ws)*self.yStd + self.yMean)
    
class FS_Regression(object):
    
    """Forward_stagewise regression:
    an easier method for getting results similar to the lasso"""
    
    def __init__(self, eps=0.01, maxIter=1000):
        self.eps = eps
        self.maxIter = maxIter  
        self.coef_ = None              
    
    def _normalize(self, X):
        xMean = np.mean(X, axis=0)
        xStd = np.std(X, axis=0)
        return (X-xMean)/xStd
    
    def _rssError(self, y, yHat):
        """returns residual sum of squares"""
        return ((y-yHat)**2).sum()
    
    def fit(self, xArr, yArr):
        xArr = self._normalize(xArr)
        self.yMean = yArr.mean()
        self.yStd = yArr.std()
        yArr = self._normalize(yArr)
            
        xMat = np.mat(xArr)        
        yMat = np.mat(yArr).T        
        m,n = np.shape(xMat)
    
        ws = np.zeros((n,1))
        returnMat = np.zeros((self.maxIter, n)) # matrix used to store weights
        for i in range(self.maxIter):
            lowestError = np.inf
            for j in range(n):
                for sign in [-1,1]:
                    wsTest = ws.copy()
                    wsTest[j] += self.eps*sign # increasing or decreasing a weight by some small amount eps
                    yTest = xMat@wsTest
                    rssE = self._rssError(np.ravel(yMat),np.ravel(yTest))
                    if rssE < lowestError:
                        lowestError = rssE
                        wsBest = wsTest
            ws = wsBest.copy()
            returnMat[i,:] = ws.T
        self.ws = np.mat(returnMat[self.maxIter-1]).T # The last estimated weights
        self.coef_ = np.ravel(self.ws)
        return ("FS_Regression(eps=%.2f, maxIter=%d)"%(self.eps,self.maxIter))
    
    def prediction(self, xArr): 
        xArr = self._normalize(xArr)                   
        xMat = np.mat(xArr)
        return (np.ravel(xMat@self.ws)*self.yStd + self.yMean)