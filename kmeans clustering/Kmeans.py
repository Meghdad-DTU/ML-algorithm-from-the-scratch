import numpy as np

class Kmeans (object):
    
    def __init__(self, k=5, method="euclidean", maxIter=300, n_init=10, normalize=True):
        self.k = k             # Number of clusters
        self.method = method
        self.maxIter = maxIter # Maximum number of iterations of the k-means algorithm for a single run  
        self.n_init = n_init   # Number of time the k-means algorithm will be run with different centroids
        self.normalize = normalize
        self.labels_ = None
        self.centroids_ = None         
                
    def _normalize(self, X):
        xMean = np.mean(X, axis=0)
        xStd = np.std(X, axis=0)
        return (X-xMean)/xStd 
    
    def _randCent(self, X):
        """Generate random centers, here we use sigma and mean to ensure it represent the whole  data""" 
        m,n = X.shape
        mean = np.mean(X, axis = 0)
        std = np.std(X, axis = 0)
        centroids = np.random.randn(self.k, n)*std + mean        
        return centroids 
        
    def _euclideanDistance(self, V1, V2):
        distance = np.sum((V1 - V2)**2, 1)**0.5
        return distance
    
    def _cosineSimilarity(self, V1, V2):
        similarity = np.sum(V1*V2,axis=1)/(np.sqrt(sum(np.power(V1.T,2))) * np.sqrt(sum(np.power(V2.T,2))))
        # If the similarity is 0, then the distance is 1.0. If the similarity is really big, then the distance falls to 0
        distance = 1/(1+similarity)
        return distance
    
    def _extendedJaccard(self, V1, V2):
        similarity = np.sum(V1*V2,axis=1)/(np.log(np.exp(sum(np.power(V1.T,2))) * np.exp(sum(np.power(V2.T,2))))-np.sum(V1*V2,axis=1))
        distance = 1/(1+similarity)
        return distance
    
    def fit(self, X):
        self.xMean = 0
        self.xStd = 1
        
        if self.normalize is True:
            self.xMean = np.mean(X, axis=0)
            self.xStd = np.std(X, axis=0)
            X = self._normalize(X)           
        
        m,n = X.shape
        if not (1 < self.k < m):
            raise Exception("number of clusters must be larger than 2 and lower than dataset size")
        
        distName = ["euclidean", "cosine", "jaccard"]
        distFun = [self._euclideanDistance, self._cosineSimilarity, self._extendedJaccard]        
                
        if self.method not in distName:
            raise Exception("There are only three types of distance: euclidean, cosine and jaccard.")
        ind = distName.index(self.method)       
        
        bestSSE = np.inf # sum of squared error: the quality of cluster assignments
        
        for i in range(self.n_init):
            centroids = self._randCent(X)    # initaial centroids        
            distances = np.zeros((m,self.k)) # distance matrix
                       
            for iteration in range(self.maxIter):
                for i in range(self.k):
                    distances[:,i] = distFun[ind]((np.tile(centroids[i,], (m,1))), X)        
           
                clusters = np.argmin(distances, axis = 1)  # assign all training data to closest center
        
                for i in range(self.k):                         # calculate mean for every cluster and update the center                    
                    centroids[i] = np.mean(X[clusters == i], axis=0)                   
        
            centroids = centroids * self.xStd + self.xMean
            
            if np.isnan(centroids).any() == True:              # in case of non value go to the next iteration
                continue                
            SSE = 0
            for i in range(self.k):
                SSE += np.sum((X[clusters==i] - centroids[i])**2)
            
            if SSE < bestSSE:
                bestSSE = SSE
                self.labels_ = clusters
                self.centroids_ = centroids                
        
        return ("Kmeans(k=%d, method=%s, maxIter=%d, n_init=%d, normalize=%r)"%(self.k,self.method,self.maxIter,self.n_init,self.normalize))
        
        

class BiKmeans (object):
    """Bisecting k-means"""
    
    def __init__(self, k=5, method="euclidean", maxIter=300, n_init=10, normalize=True):
        self.k = k             # Number of clusters
        self.method = method
        self.maxIter = maxIter # Maximum number of iterations of the k-means algorithm for a single run  
        self.n_init = n_init   # Number of time the k-means algorithm will be run with different centroids
        self.normalize = normalize         
        
    def fit(self, X):
        self.X = X       
        m,n = X.shape 
        self.centroids_ = np.zeros((self.k, n)) 
        self.labels_ = np.zeros(m) + (self.k - 1) #(k-1) is the label for the last remainig cluster 
        
        noCent = 0            # Number of initial ceters         
        while noCent < self.k-1 :
            m,n = X.shape
            model = Kmeans(k=2, method=self.method, maxIter=self.maxIter, n_init=self.n_init, normalize=self.normalize)
            model.fit(X)    
            centroids = model.centroids_
            clusters = model.labels_  
                        
            SSE0 = np.sum((X[clusters==0] - centroids[0])**2)
            SSE1 = np.sum((X[clusters==1] - centroids[1])**2)            
            if SSE0 < SSE1:
                notSplit = 0               
            else:
                notSplit = 1
            
            self.centroids_[noCent] = centroids[notSplit]
            self.centroids_[noCent+1] = centroids[1-notSplit]
            
            # method 1: finding labels
            ab,a_ind,b_ind = np.intersect1d(np.sum(self.X,axis=1),np.sum(X[clusters == notSplit],axis=1), return_indices=True)
            for x in a_ind: 
                self.labels_[x] = noCent                
            """# method 2:
            for i, x in enumerate(self.X):
                if np.equal(X[clusters == notSplit],x).any():
                    self.labels_[i] = noCent"""
                
            X = X[clusters != notSplit]                
            noCent+=1
            
        return ("BiKmeans(k=%d, method=%s, maxIter=%d, n_init=%d, normalize=%r)"%(self.k,self.method,self.maxIter,self.n_init,self.normalize))        
      
