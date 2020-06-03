class Knn:
    def __init__(self, k):
        self.k = k        
        self.pred_ = None
        self.X_ = None
        self.y_ = None
        self.distance_= None
        
    def _euclideanDistance(self, V1, V2):
        return np.sum((V1 - V2)**2, 1)**0.5
    
    def _cosineSimilarity(self, V1, V2):
        similarity = np.sum(V1*V2,axis=1)/(np.sqrt(sum(np.power(V1.T,2))) * np.sqrt(sum(np.power(V2.T,2))))
        return similarity
    
    def _extendedJaccard(self, V1, V2):
        similarity = np.sum(V1*V2,axis=1)/(np.log(np.exp(sum(np.power(V1.T,2))) * np.exp(sum(np.power(V2.T,2))))-np.sum(V1*V2,axis=1))
        return similarity
                                
    def fit(self, X, y, distance="euclidean"):
        if type(y)==list:
            y = np.array(y)
        ## for prediction method  
        if y is not None:
            self.y_ = y
            self.X_ = X
            self.distance_ = distance
            self.sizeX = X.shape
            
        distanceFunction = [self._euclideanDistance, self._cosineSimilarity, self._extendedJaccard]
        if not (0 < self.k < self.sizeX[0]):
            raise Exception("k must be lower than dataset size")        
        pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            if distance=="euclidean":
                distances = distanceFunction[0](np.tile(X[i,], (self.sizeX[0],1)), self.X_)
                sortedDistIndicies = distances.argsort()[:self.k]
            elif distance=="cosine":
                distances = distanceFunction[1](np.tile(X[i,], (self.sizeX[0],1)), self.X_)
                sortedDistIndicies = distances.argsort()[::-1][:self.k]
            elif distance=="jaccard":
                distances = distanceFunction[2](np.tile(X[i,], (self.sizeX[0],1)), self.X_)
                sortedDistIndicies = distances.argsort()[::-1][:self.k]
            else :
                raise Exception("There are only three types of distance: euclidean, cosine and jaccard.")
            
            voteIlabel = self.y_[sortedDistIndicies]
            classCount={}            
            for vote in voteIlabel:                
                classCount[vote] = classCount.get(vote,0) + 1
            sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1), reverse=True)
            pred[i] = sortedClassCount[0][0]
        self.pred_= pred
        return("Knn(k=%d, distance=%s)"%(self.k, distance))
    
    def score(self):
        return [pred==true for pred,true in zip(self.pred_, self.y_)].count(True)/len(self.y_)
    
    def prediction(self,X):
        self.fit(X, y= None, distance= self.distance_)
        return self.pred_