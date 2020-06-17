import numpy as np
from scipy.stats import norm
import statistics as stats 

class NB_DiscreteClassifier(object):
    
    def __init__(self):
        self.X_= None
        self.y = None
        self.probabilityTable_ = None        
        self.prediction_ = None        
    
    def _uniqValue (self, X, axis):
        """returns a dictionary including all unique values of an axis of X"""
        myDict = dict()
        for x in X[:,axis]:
            myDict[x] = myDict.get(x, 0) + 1
        return myDict
    
    def _split(self, X, y, classLabel):
        """splits X based on an unique value of y"""
        index = np.where(y == classLabel)[0]
        return X[index,:] 
    
    def fit(self, X, y):
        assert  min(y) == 0        
        self.X_ = X
        self.y_ = y
        classLabels = list(set(y))                        ## unique values of the target variable
        probabilityTable = {}                             ## probability table for all axis of X conditioned on y
        for classLabel in classLabels:
            probabilityTable[classLabel] = {}
            Xsplit = self._split(X, y, classLabel)
            total = Xsplit.shape[0]
            for axis in range(Xsplit.shape[1]):
                dic = self._uniqValue(Xsplit, axis)
                prbability = {k: v / total for k, v in dic.items()}    ## calculate probaility
                probabilityTable[classLabel][axis]= prbability
        self.probabilityTable_ = probabilityTable            
        return "NaiveBayesDiscreteClassifier()"
    
    def score(self, X = None):
        """ provides the accuracy of trained model """
        total = len(self.y_)
        dic = self._uniqValue(self.y_.reshape(-1,1),0)
        classPrbability = {k: v / total for k, v in dic.items()} 
        if X is None :
            matProbability = np.zeros((self.X_.shape[0], len(set(self.y_)))) 
        else:
            matProbability = np.zeros((X.shape[0], len(set(self.y_))))
            self.X_ = X
            
        i=0
        for row in self.X_:
            for classLabel in classPrbability.keys():    
                probNewX = np.zeros(self.X_.shape[1])
                for newX, axis in zip(row, range(self.X_.shape[1])):
                    if newX in self.probabilityTable_[classLabel][axis].keys():
                        probNewX[axis] = self.probabilityTable_[classLabel][axis][newX]
                    else:
                        probNewX[axis] = 0.0000002                      ## low probability instead of zero
                ## Using log-probabilities for Naive Bayes: 
                # matProbability[i, classLabel] = np.prod(probNewX[axis]) * classPrbability[classLabel]                
                matProbability[i, classLabel] = np.sum(np.log(probNewX[axis])) + np.log(classPrbability[classLabel])
            i+=1
            
            pred = matProbability.argsort()[:,len(set(self.y_))-1]      ## returns the indices of the sorted array
            self.prediction_ = pred
            accuracy = [pred==true for pred,true in zip(pred, self.y_)].count(True)/len(self.y_)
        return accuracy 
    
    def prediction(self, X):
        self.score(X = X)
        return self.prediction_
        

 
class NB_GuassianClassifier(object):
    
    def __init__(self):
        self.X_= None
        self.y = None
        self.normParameters_ = None
        self.prediction_ = None        
    
    def _split(self, X, y, classLabel):
        index = np.where(y == classLabel)[0]
        return X[index,:] 
    
    def _uniqValue (self, X, axis):
        myDict = dict()
        for x in X[:,axis]:
            myDict[x] = myDict.get(x, 0) + 1
        return myDict
    
    def fit(self, X, y):
        assert  min(y) == 0
        self.X_ = X
        self.y_ = y
        classLabels = list(set(y))
        normParameters = {}
        for classLabel in classLabels:
            normParameters[classLabel] = {}
            Xsplit = self._split(X, y, classLabel)
            XsplitMean = Xsplit.mean(axis=0)
            XsplitStd = Xsplit.std(axis=0, ddof=1)
            normParameters[classLabel]["mean"] = XsplitMean
            normParameters[classLabel]["std"] = XsplitStd            
        self.normParameters_ = normParameters           
        return "NaiveBayesGuassianClassifier()"
    
    def score(self, X = None):
        total = len(self.y_)
        dic = self._uniqValue(self.y_.reshape(-1,1),0)
        classPrbability = {k: v / total for k, v in dic.items()} 
        if X is None :
            matProbability = np.zeros((self.X_.shape[0], len(dic)))
        else:
            matProbability = np.zeros((X.shape[0], len(dic)))
            self.X_ = X            
        
        for classLabel,_ in dic.items():
            mean = np.tile(self.normParameters_[classLabel]["mean"], (self.X_.shape[0],1 ))
            std = np.tile(self.normParameters_[classLabel]["std"], (self.X_.shape[0],1 ))
            calProbability = norm.pdf(self.X_, mean, std)
            ## Using log-probabilities for Naive Bayes: 
            # matProbability[:,classLabel] = np.prod(calProbability,1) * classPrbability[classLabel]
            matProbability[:,classLabel] = np.sum(np.log(calProbability),1) + np.log(classPrbability[classLabel])
        pred = matProbability.argsort()[:,len(dic)-1]
        self.prediction_ = pred
        accuracy = [pred==true for pred,true in zip(pred, self.y_)].count(True)/total
        return accuracy 
    
    def prediction(self, X):
        self.score(X = X)
        return self.prediction_









        