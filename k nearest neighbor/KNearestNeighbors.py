import numpy as np
import operator

class KnnClassifier(object):
    
    def __init__(self, k: int):
        self.k = k        
                
    def _euclideanDist(self, V1, V2):
        distance = np.sum((V1 - V2)**2, 1)**0.5
        return distance
    
    def _cosineDist(self, V1, V2):
        similarity = np.sum(V1*V2,axis=1)/(np.sqrt(sum(np.power(V1.T,2))) * np.sqrt(sum(np.power(V2.T,2))))
        distance = 1/(1+similarity)
        return distance
    
    def _extendedJaccardDist(self, V1, V2):
        similarity = np.sum(V1*V2,axis=1)/(np.log(np.exp(sum(np.power(V1.T,2))) * np.exp(sum(np.power(V2.T,2))))-np.sum(V1*V2,axis=1))
        distance = 1/(1+similarity)
        return distance
                                
    def _check_X_y(self, X, y):
        """Validate assumptions about format of input data"""
        assert type(X) and type(y) == np.ndarray, 'Expecting the type of input data to be array'
        return X, y            
    
    def fit(self,  X: np.ndarray, y: np.ndarray, distance="euclidean"):        
        ## for prediction purpose         
        if y is not None:
            X, y = self._check_X_y(X, y)
            self.y_ = y
            self.X_ = X
            self.distance_ = distance
            self.sizeX = X.shape            
        
        if not (0 < self.k < self.sizeX[0]):
            raise Exception("k must be lower than dataset size")        
        
        pred = np.zeros(X.shape[0])        
        
        distNameList = ["euclidean","cosine","jaccard"]
        distFunctionList = [self._euclideanDist, self._cosineDist, self._extendedJaccardDist]
        
        if self.distance_ not in distNameList:
            raise Exception("There are only three types of distance: euclidean, cosine & jaccard.")
        
        ind = distNameList.index(distance)              
        for i in range(X.shape[0]):
            distances = distFunctionList[ind](np.tile(X[i,], (self.sizeX[0],1)), self.X_)
            sortedDistIndicies = distances.argsort()[:self.k]           
            
            voteIlabel = self.y_[sortedDistIndicies]
            classCount={}            
            for vote in voteIlabel:                
                classCount[vote] = classCount.get(vote,0) + 1
            sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1), reverse=True)
            pred[i] = sortedClassCount[0][0]
        self.pred_= pred
        return("Knn Classifier(k=%d, distance=%s)"%(self.k, self.distance_))
    
    def score(self):
        return [pred==true for pred,true in zip(self.pred_, self.y_)].count(True)/len(self.y_)
    
    def prediction(self, X: np.ndarray):
        self.fit(X, y= None, distance= self.distance_)
        return self.pred_
        
class KnnRegressor(object):
    
    def __init__(self, k: int):
        self.k = k        
                
    def _euclideanDist(self, V1, V2):
        distance = np.sum((V1 - V2)**2, 1)**0.5
        return distance
    
    def _cosineDist(self, V1, V2):
        similarity = np.sum(V1*V2,axis=1)/(np.sqrt(sum(np.power(V1.T,2))) * np.sqrt(sum(np.power(V2.T,2))))
        distance = 1/(1+similarity)
        return distance
    
    def _extendedJaccardDist(self, V1, V2):
        similarity = np.sum(V1*V2,axis=1)/(np.log(np.exp(sum(np.power(V1.T,2))) * np.exp(sum(np.power(V2.T,2))))-np.sum(V1*V2,axis=1))
        distance = 1/(1+similarity)
        return distance
                                
    def _check_X_y(self, X, y):
        """Validate assumptions about format of input data"""
        assert type(X) and type(y) == np.ndarray, 'Expecting the type of input data to be array'
        return X, y            
    
    def fit(self,  X: np.ndarray, y: np.ndarray, distance="euclidean"):        
        ## for prediction method         
        if y is not None:
            X, y = self._check_X_y(X, y)
            self.y_ = y
            self.X_ = X
            self.distance_ = distance
            self.sizeX = X.shape            
        
        if not (0 < self.k < self.sizeX[0]):
            raise Exception("k must be lower than dataset size")        
        
        pred = np.zeros(X.shape[0])        
        
        distNameList = ["euclidean","cosine","jaccard"]
        distFunctionList = [self._euclideanDist, self._cosineDist, self._extendedJaccardDist]
        
        if self.distance_ not in distNameList:
            raise Exception("There are only three types of distance: euclidean, cosine & jaccard.")
        
        ind = distNameList.index(distance)              
        for i in range(X.shape[0]):
            distances = distFunctionList[ind](np.tile(X[i,], (self.sizeX[0],1)), self.X_)
            sortedDistIndicies = distances.argsort()[:self.k]           
            
            yNeighbor = self.y_[sortedDistIndicies]
            pred[i] = np.mean(yNeighbor)
        self.pred_= pred
        return("Knn Regressor(k=%d, distance=%s)"%(self.k, self.distance_))
    
    def score(self):
        """Return the coefficient of determination R^2 of the prediction"""
        trues = self.y_
        predicted = self.pred_
        r2=max(0,1-np.sum((trues-predicted)**2)/np.sum((trues-np.mean(trues))**2))
        return r2
    
    def prediction(self, X: np.ndarray):
        self.fit(X, y= None, distance= self.distance_)
        return self.pred_