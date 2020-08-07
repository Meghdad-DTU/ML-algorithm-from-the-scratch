import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PCA(object):
    """Principal Component Analysis"""
    
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
    
    def fit_transform(self, X):
        """fit model and apply the dimensionality reduction on X"""
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
        plt.ylabel('Percentage of Variance Explained')
        plt.xlabel('Principal Component')
        plt.xticks(rotation=45) 
        plt.title('Scree Plot')
        plt.grid(alpha=.5,linestyle='--',axis="y")
        plt.show()  
        
class SVD(object):
    """Singular Value Decomposition"""
    
    def __init__(self, n_sv  = None):
        self.n_sv = n_sv # Number of singular values
        
    def _check_X(self, X):
        """ Validate assumptions about format of input data"""
        assert type(X) == np.ndarray, 'Expecting the type of input data to be array'
        return X 
    
    def fit(self, X):
        """fit model to X"""
        X = self._check_X(X)
        self.X = X
        m,n = X.shape
        if self.n_sv is None:
            self.n_sv = n            
        assert self.n_sv <= n, 'Number of singular values is larger than the number of features' 
        
        U,sigma,Vt = np.linalg.svd(X)
        self.U_ = U[:,0:self.n_sv]      
        self.Vt_ = Vt[:self.n_sv,:]
        self.singular_values_ = sigma[0:self.n_sv]       
        self.sigma_ = np.diag(sigma[0:self.n_sv])
        self.explained_variance_ = np.power(sigma,2)[:self.n_sv]
        self.explained_variance_ratio_ = (np.power(sigma,2)/np.sum(np.power(sigma,2)))[:self.n_sv] 
        # For scree plot
        self.expVarRatio = (np.power(sigma,2)/np.sum(np.power(sigma,2)))        
        return "SVD(n_sv=%d)"%self.n_sv
    
    def fit_transform(self, X):
        """fit model to X and perform dimensionality reduction on X"""           
        self.fit(X)
        reducedMat = self.U_@self.sigma_
        return reducedMat
   
    def inv_transform(self,X):
        """reconstructing an approximation of the original matrix"""        
        return self.fit_transform(X)@self.Vt_   
        
        
    def screePlot(self, number_SV = None):
        """
        plot the explained variance by selected singular values using barplot        
        number_SV: Number of SV for displaying on the scree plot         
        """
        self.fit(self.X)
        if number_SV is None:
            number_SV = self.X.shape[1]
        percent_variance = np.round(self.expVarRatio*100, decimals =2)[:number_SV]
        accSum = np.cumsum(percent_variance)
        columns = ["SV%d"%(i+1) for i in range(0,number_SV)]
        plt.figure(figsize=(4, 6))
        plt.bar(x= range(0,number_SV), height=accSum, tick_label=columns, width =0.4, alpha = 0.4)
        plt.plot(range(0,number_SV),percent_variance)
        plt.scatter(range(0,number_SV),percent_variance, color="r")
        plt.ylabel('Percentage of Variance Explained')
        plt.xlabel('Singular Value')
        plt.xticks(rotation=45) 
        plt.title('Scree Plot')
        plt.grid(alpha=.5,linestyle='--',axis="y")
        plt.show()