import numpy as np
from math import log

class DecisionTreeClassifier(object):
        
    def __init__(self, max_depth, min_size, criterion="gini"):
        self.max_depth = max_depth
        self.min_size = min_size
        self.criterion = criterion
        self.trees_= None
                
    def _majorityCount(self,y):
        """Return the class that occurs with the greatest frequency."""
        uniqueList = set(y)
        sortedClassCount = sorted(y,reverse = True)
        return sortedClassCount[0]
       
    def _calcShannonEntropy(self, y):
        """Shannon entropy calculation"""
        total = len(y)
        dic = {}
        for item in y:
            dic[item] = dic.get(item,0) + 1
        dic = {k: v / total for k, v in dic.items()} # calculate probaility
        shannonEnt = 0.0
        for k, prob in dic.items():
            shannonEnt += prob * log(prob,2)        
        return (-1*shannonEnt) 
    
    def _calcGiniScore(self, y):
        """Gini score calculation"""
        total = len(y)
        dic = {}
        for item in y:
            dic[item] = dic.get(item,0) + 1
        dic = {k: v / total for k, v in dic.items()} # calculate probaility
        gini = 0.0
        for k, prob in dic.items():
            gini += prob**2        
        return (1-gini) 
    
    def _splitDataSet(self, X, y, axis, value):
        """
        Dataset splitting on a given feature and its values:
        X: Feature set
        y: target variable
        axis: a given feature
        value: existing value within the axis 
        
        """
        if type(y) == list:
            y = np.array(y)
        y = y.flatten().reshape(-1,1)
        assert y.shape[0] == X.shape[0]
        dataSet = np.hstack((X,y))
        # separate dataSet into 2 groups
        left_index, right_index  = np.where(dataSet[:,axis] < value)[0], np.where(dataSet[:,axis] >= value)[0]     
        left_subSet  = dataSet[left_index,:]
        right_subSet = dataSet[right_index,:]       
        return left_subSet, right_subSet
    
    def _chooseBestFeatureToSplit(self, X, y):
        """Choosing the best feature to split on"""
        noColumns = X.shape[1] 
        finalCrt = 1  
        calCriteria = [self._calcGiniScore, self._calcShannonEntropy]
        if self.criterion == "gini":
            function = calCriteria[0]
        elif self.criterion == "entropy":
            function = calCriteria[1]
        else:
            raise Exception("Unknown criterion type: %s. Valid criteria are: gini and entropy." % (criterion))
        for axis in range(noColumns): # iterating through each feature
            uniqVal = {}
            for x in X[:,axis]:
                uniqVal[x] = uniqVal.get(x,0) + 1  
            for key, value in uniqVal.items():
                left_subSet, right_subSet = self._splitDataSet(X, y, axis, key)
                if len(left_subSet) == 0: 
                    probLeft = 0 
                    crtLeft = 0
                else :
                    probLeft = len(left_subSet[:,-1])/float(len(y))
                    crtLeft = probLeft * function(left_subSet[:,-1])
                if len(right_subSet) == 0: 
                    probRight = 0
                    crtRight = 0
                else :
                    probRight = len(right_subSet[:,-1])/float(len(y))
                    crtRight = probRight * function(right_subSet[:,-1])
                crt = crtLeft + crtRight
                if crt < finalCrt:
                    finalCrt = crt
                    col = axis
                    cutOff = key
        return col, cutOff 
                
    def tree(self, X, y, label=None, par_node = {}, depth = 0):    
        """
        recursively Tree-building:
        X: Feature set
        y: target variable
        par_node: will be the tree generated for this X and y. 
        depth: the depth of the current layer
        
        """
        
        if par_node is None:                            # base case 1: tree stops at previous level
            return None
        elif X.shape[0] <= self.min_size:               # base case 2: No.of data in this group is smaller than min_size
            return {'val': self._majorityCount(y)}
        elif len(sorted(set(y))) == 1:                  # base case 3: all y is the same in this group
            return {'val':y[0]}
        elif depth >= self.max_depth:                   # base case 4: max depth reached 
            return None
        else:   # Recursively generate trees! 
        # find one split given an information gain 
            col, cutOff = self._chooseBestFeatureToSplit(X, y)       
            left_subSet, right_subSet = self._splitDataSet(X, y, col, cutOff)
            y_left, X_left = left_subSet[:,-1], left_subSet[:,:-1]
            y_right, X_right = right_subSet[:,-1], right_subSet[:,:-1]
        
            if label is not None:
                bestFeat = label[col]
            else:
                bestFeat = col
            
            par_node = {'col': bestFeat,
                        'index_col': col,
                        'cut_off':cutOff,
                        'samples':len(y),                    
                        'val': self._majorityCount(y)}  # save the information 
        
            # generate tree for the left hand side data
            par_node['left'] = self.tree(X_left, y_left, label, {}, depth+1)   
            # right hand side trees
            par_node['right'] = self.tree(X_right, y_right, label, {}, depth+1)  
            depth += 1   # increase the depth since we call fit once            
            return par_node
        
    def fit(self, X, y, label = None):
        self.trees_ = self.tree(X, y, label, par_node = {}, depth = 0)
        return("DecisionTreeClassifier(max_depth=%d, min_size=%d, criterion=%s)"%
              (self.max_depth,self.min_size,self.criterion))
        
    def _singleRowPrediction(self, row):        
        """Predict class for a single sample."""
        trees = self.trees_
        while trees.get('cut_off'):   # if not leaf node
            if row[trees['index_col']] < trees['cut_off']:   # get the direction 
                trees = trees['left']
            else: 
                trees = trees['right']        
        else:   # if leaf node, return value
            return trees.get('val')
            
    def prediction(self, X):
        if type(X) == list:
            X = np.array(X)
        if X.ndim ==1:
            pred = self._singleRowPrediction(X)
        else:
            pred = np.zeros(len(X))
            for i, row in enumerate(X):
                pred[i] = self._singleRowPrediction((row))
        return pred