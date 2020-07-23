import numpy as np

class DecisionTreeClassifier(object):
        
    def __init__(self, max_depth= 30, min_size=3, criterion="gini"):
        self.max_depth = max_depth  ## The maximum depth of the tree
        self.min_size = min_size    ## The minimum number of samples required to split an internal node
        self.criterion = criterion
        self.depth_ = 0             ## final depth of the tree
        self.predicted = None
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
            shannonEnt += prob * np.log(prob,2)        
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
    
    def _check_X_y(self, X, y):
        """ Validate assumptions about format of input data"""
        assert type(X) and type(y) == np.ndarray, 'Expecting the type of input data to be array'
        return X, y 
    
    def _splitDataSet(self, dataSet, axis, value):
        """
        Dataset splitting on a given feature and its values:
        axis: a given feature
        value: existing value within the axis 
        
        """       
        # separate dataSet into 2 groups
        left_index, right_index  = np.where(dataSet[:,axis] < value)[0], np.where(dataSet[:,axis] >= value)[0]     
        left_subSet  = dataSet[left_index,:]
        right_subSet = dataSet[right_index,:]       
        return left_subSet, right_subSet
    
    def _chooseBestFeatureToSplit(self, dataSet):
        """Choosing the best feature to split on"""
        m,n = dataSet.shape 
        finalCrt = np.inf  
        calCriteria = [self._calcGiniScore, self._calcShannonEntropy]
        if self.criterion == "gini":
            function = calCriteria[0]
        elif self.criterion == "entropy":
            function = calCriteria[1]
        else:
            raise Exception("Unknown criterion type: %s. Valid criteria are: gini and entropy." % (criterion))
        for axis in range(n-1):              # iterating through each feature
            for val in set(dataSet[:,axis]): # iterating through each unique value of the feature
                left_subSet, right_subSet = self._splitDataSet(dataSet, axis, val)               
                if len(left_subSet) == 0: 
                    probLeft = 0 
                    crtLeft = 0
                else :
                    probLeft = len(left_subSet[:,-1])/float(len(dataSet[:,-1])) 
                    crtLeft = probLeft * function(left_subSet[:,-1])
                if len(right_subSet) == 0: 
                    probRight = 0
                    crtRight = 0
                else :
                    probRight = len(right_subSet[:,-1])/float(len(dataSet[:,-1]))
                    crtRight = probRight * function(right_subSet[:,-1])                
                
                crt = crtLeft + crtRight
                if crt < finalCrt:
                    finalCrt = crt
                    bestAxis = axis
                    bestValue = val
                
        return bestAxis, bestValue 
                
    def tree(self, dataSet, label=None, par_node = {}, depth = 0):    
        """
        recursively Tree-building:
        par_node: will be the tree generated for dataSet. 
        depth: the depth of the current layer
        
        """
        
        if par_node is None:                            # base case 1: tree stops at previous level
            return None
        elif dataSet.shape[0] < self.min_size:         # base case 2: No.of data in this group is smaller than min_size
            return {'val': self._majorityCount(dataSet[:,-1])}
        elif len((set(dataSet[:,-1]))) == 1:            # base case 3: all y is the same in this group
            return {'val':dataSet[:,-1][0]}
        elif depth >= self.max_depth:                   # base case 4: max depth reached 
            return None
        else:   # Recursively generate trees! 
        # find one split given an information gain 
            col, cutOff = self._chooseBestFeatureToSplit(dataSet)       
            left_subSet, right_subSet = self._splitDataSet(dataSet, col, cutOff)
                   
            if label is not None:
                bestFeat = label[col]
            else:
                bestFeat = col
            
            par_node = {'col': bestFeat,
                        'index_col': col,
                        'cut_off':cutOff,
                        'samples':len(dataSet[:,-1]),                    
                        'val': self._majorityCount(dataSet[:,-1])}  # save the information 
        
            # generate tree for the left hand & right hand sides 
            par_node['left'] = self.tree(left_subSet, label, {}, depth+1)            
            par_node['right'] = self.tree(right_subSet, label, {}, depth+1)  
            depth += 1   # increase the depth since we call fit once 
            
            if depth > self.depth_:
                self.depth_ = depth 
            return par_node
        
    def fit(self, X: np.ndarray, y: np.ndarray, label = None):
        X, y = self._check_X_y(X, y)
        self.X = X
        self.y = y
        dataSet = np.hstack((X, y.reshape(-1,1))) 
        self.trees_ = self.tree(dataSet, label, par_node = {}, depth = 0)
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
            
    def score(self,X=None):
        """Return the mean accuracy on the given test data and labels"""
        if X is not None:
            self.X = X
        if self.X.ndim ==1:
            pred = self._singleRowPrediction(self.X)
        else:
            yPred = np.zeros(len(self.X))
            for i, row in enumerate(self.X):
                yPred[i] = self._singleRowPrediction((row))
            self.predicted = yPred
        return [pred==true for pred,true in zip(yPred, self.y)].count(True)/len(self.y)
    
    def prediction(self, X):
        """Predict class for X"""
        self.score(X)        
        return self.predicted