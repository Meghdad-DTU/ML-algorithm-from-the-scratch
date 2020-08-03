import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PCA(object):
    
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.explained_variance_ = None         
        self.explained_variance_ratio_ = None   
    
    def _check_X(self, X):
        """ Validate assumptions about format of input data"""
        assert type(X) == np.ndarray, 'Expecting the type of input data to be array'
        return X    
    
    def fit(self, X): 
        X = self._check_X(X)
        self.X = X
        m,n = X.shape
        if self.n_components is None:
            self.n_components = n            
        assert self.n_components <= n, 'Number of components is larger than number of features'       
        
        # Because of calculating the center of data, shifting them and centering data on the orginal 
        xMean = np.mean(X, axis=0)
        X = X - xMean             
        covX = np.cov(X, rowvar=0)        
        eigenVals, eigenVects = np.linalg.eig(covX) 
        eigenValsSort = np.sort(eigenVals)[::-1]
        
        self.expVarRatio = eigenValsSort/np.sum(eigenValsSort)  
                
        # The amount of variance explained by each of the selected components
        self.explained_variance_ = eigenValsSort[:self.n_components]
        # Percentage of variance explained by each of the selected components"""
        self.explained_variance_ratio_ = (eigenValsSort/np.sum(eigenValsSort))[:self.n_components]               
        
        # Sort n number of the eigenvalues from largest to smallest
        eigenValsInd = eigenVals.argsort()[::-1][:self.n_components]
        # Reduced eigen vectors
        redEigVects = eigenVects[:,eigenValsInd]        
        self.components_ = redEigVects
        
        # Transform data into new dimensions
        redDim = X@redEigVects
        self.transform_ = redDim
        self.inv_transform_ = (redDim@redEigVects.T) + xMean
              
        return ("PCA(n_components=%d)"%(self.n_components))   
    
    def transform(self, X):
        """apply the dimensionality reduction on X"""
        self.fit(X)
        return self.transform_
    
    def inv_transform(self, X):
        """transform reduced data back to its original space"""
        self.fit(X)
        return self.inv_transform_
    
    def loadingMatrix(self):
        "calculate the Pearson correlation between principal component and variable"
        _, n = self.X.shape  
        loadMat = np.zeros((n,self.n_components))
        for i, pc in enumerate(self.transform_.T):
            for j, col in enumerate(self.X.T):
                loadMat[j,i] = np.corrcoef(pc,col)[0,1]        
        return loadMat   
    
    def scoreCoefMatrix(self):
        "component score coefficient Matrix"
        _, n = self.X.shape  
        loadMat = np.zeros((n,self.n_components))
        for i, pc in enumerate(self.transform_.T):
            for j, col in enumerate(self.X.T):
                loadMat[j,i] = np.corrcoef(pc,col)[0,1]        
        scoreCoefMat = np.linalg.inv(np.corrcoef(self.X,rowvar=0))@loadMat  
        return scoreCoefMat     
    
    def screePlot(self, number_PC = None):
        """
        plot the explained variance by selected components using barplot        
        number_PC: Number of PC for displaying on the scree plot         
        """
        self.fit(self.X)
        if number_PC is None:
            number_PC = self.X.shape[1]
        percent_variance = np.round(self.expVarRatio*100, decimals =2)[:number_PC]
        accSum = np.cumsum(percent_variance)
        columns = ["PC%d"%(i+1) for i in range(0,number_PC)]
        plt.figure(figsize=(4, 6))
        plt.bar(x= range(0,number_PC), height=accSum, tick_label=columns, width =0.4, alpha = 0.4)
        plt.plot(range(0,number_PC),percent_variance)
        plt.scatter(range(0,number_PC),percent_variance, color="r")
        plt.ylabel('Percentate of Variance Explained')
        plt.xlabel('Principal Component')
        plt.xticks(rotation=45) 
        plt.title('PCA Scree Plot')
        plt.grid(alpha=.5,linestyle='--',axis="y")
        plt.show()  