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


class DecisionTreeRegressor(object):
        
    def __init__(self, max_depth= 30, min_size=3, tol_error=10, criterion="rege"):
        self.max_depth = max_depth  ## maximum depth of the tree
        self.min_size = min_size    ## minimum data instances to include in a split 
        self.tol_error = tol_error  ## tolerance on the error reduction: pre-pruning
        self.criterion = criterion 
        self.depth_ = 0             ## final depth of the tree
        self.predicted = None
        self.trees_= None
        
    def _regLeaf(self, y):
        """returns mean of the target value"""
        return y.mean()
    
    def _regErr(self, y):
        """returns total squared error""" 
        return y.var() * len(y) 
    
    def _maErr(self, y):
        """returns mean absolute error""" 
        return np.mean(np.abs(y-y.mean()))* len(y) 
    
    def _msErr(self, y):
        """returns mean squared error""" 
        return np.mean((y-y.mean())**2) * len(y)   
    
    def _check_X_y(self, X, y):
        """validate assumptions about format of input data"""
        assert type(X) and type(y) == np.ndarray, 'Expecting the type of input data to be array'
        return X, y 
    
    def _binSplitDataSet(self, dataSet, axis, value):
        """
        separate dataSet into 2 groups on a given feature and its values
        axis: a given feature
        value: existing value within the axis        
        """
        
        left_index, right_index  = np.where(dataSet[:,axis] < value)[0], np.where(dataSet[:,axis] >= value)[0]     
        left_subSet  = dataSet[left_index,:]
        right_subSet = dataSet[right_index,:]       
        return left_subSet, right_subSet
    
    def _chooseBestFeatureToSplit(self, dataSet):
        """returns the best axis and its cut off for binary split"""
        crtFun = [self._regErr, self._maErr, self._msErr] 
        crtName = ["rege", "mae", "mse"]
        
        if self.criterion in crtName:
            ind = crtName.index(self.criterion)
            qosFun = crtFun[ind]          # quality of split functions
        else:
            raise Exception("Unknown criterion %s. Valid names are: rege, mae & mse."%self.criterion)            
        
        if len(set(dataSet[:,-1])) == 1: # exit if all values the target variable are equal
            return None,  self._regLeaf(dataSet[:,-1])
   
        m,n = dataSet.shape
        S = qosFun(dataSet[:,-1])
        bestS = np.inf
        
        for axis in range(n-1):              # iterating through each feature
            for val in set(dataSet[:,axis]): # iterating through each unique value of the feature
                left_subSet, right_subSet = self._binSplitDataSet(dataSet, axis, val)
                if (left_subSet.shape[0] < self.min_size) or (right_subSet.shape[0] < self.min_size): 
                    continue
                newS = qosFun(left_subSet[:,-1]) + qosFun(right_subSet[:,-1])
                if newS < bestS:
                    bestAxis = axis
                    bestValue = val
                    bestS = newS
        ## important: 
        # Exit if low error reduction, it returns None, the mean of leaf node instead of bestAxis, bestValue   
        if (S - bestS) < self.tol_error:  
            return None, self._regLeaf(dataSet[:,-1]) 
        ## important: 
        # Exit if split creates small dataset, it returns None, the mean of leaf node instead of bestAxis, bestValue
        left_subSet, right_subSet = self._binSplitDataSet(dataSet, bestAxis, bestValue)
        if (left_subSet.shape[0] < self.min_size) or (right_subSet.shape[0] < self.min_size): 
            return None, self._regLeaf(dataSet[:,-1])
        return bestAxis, bestValue

    def tree(self, dataSet, label = None, par_node = {}, depth = 0):   
        axis, val = self._chooseBestFeatureToSplit(dataSet) 
        
        if depth >= self.max_depth:                       
            return val
        else:            
            if axis == None: 
                return val           
            
            if label is not None:
                bestFeat = label[axis]
            else:
                bestFeat = axis
            
            par_node = {'col': bestFeat,
                        'index_col': axis,
                        'cut_off':val,
                        'samples':len(dataSet[:,-1])}  # no of instances in each split        
          
            left_subSet, right_subSet = self._binSplitDataSet(dataSet, axis, val)
            # generate tree for the left and right hand side data  
            par_node['left'] = self.tree(left_subSet, label, {}, depth+1)
            par_node['right'] = self.tree(right_subSet, label, {}, depth+1)
            depth += 1
            
            if depth > self.depth_:
                self.depth_ = depth                  
        return par_node
    
    def fit(self, X: np.ndarray, y: np.ndarray, label = None):
        X, y = self._check_X_y(X, y)
        self.X = X
        dataSet = np.hstack((X, y.reshape(-1,1))) 
        self.trees_ = self.tree(dataSet, label, par_node = {}, depth = 0)
        return("DecisionTreeRegressor(max_depth=%d, min_size=%d, tol_error=%d, criterion=%s)"%
              (self.max_depth,self.min_size,self.tol_error,self.criterion))
    
    def _isTree(self, obj):
        """returns Boolean by testing if a variable is a tree"""
        #return (type(obj)==dict)
        return (type(obj).__name__ =="dict")

    def _singleRowPrediction(self, row):        
        """Predict value for a single sample."""
        trees = self.trees_
        while self._isTree(trees):                           # while it is not a leaf node                                 
            if row[trees['index_col']] < trees['cut_off']:   # get the direction 
                trees = trees['left']
            else: 
                trees = trees['right']        
        else:   # if leaf node, return value
            return trees
            
    def prediction(self, X=None):
        if X is not None:
            self.X = X
        if self.X.ndim ==1:
            pred = self._singleRowPrediction(self.X)
        else:
            yPred = np.zeros(len(self.X))
            for i, row in enumerate(self.X):
                yPred[i] = self._singleRowPrediction((row))            
        return yPred 
    
    
    def _getMean(self, tree):
        """
        recursive function that descends a tree until it hits only leaf nodes. 
        When it finds two leaf nodes, it takes the average of these two nodes.
        tree: built tree from training data (i.e. self.trees_)
        """
        
        if self._isTree(tree['right']): 
            tree['right'] = self._getMean(tree['right'])
        if self._isTree(tree['left']): 
            tree['left'] = self._getMean(tree['left'])
        return (tree['left']+tree['right'])/2.0
    
    def prune(self, tree, testData):
        """
        recursive function for post-pruning: it returns a pruned tree
        - Split the test data for the given tree
          -- If the either split is a tree: call prune on that split
          -- Calculate the error associated with merging two leaf nodes
          -- Calculate the error without merging
          -- If merging results in lower error then merge the leaf nodes

        """
        if testData.shape[0] == 0:  # check to see if the test data is empty
            return self._getMean(tree)
        
        if (self._isTree(tree['right']) or self._isTree(tree['left'])):
            left_subSet, right_subSet = self._binSplitDataSet(testData, tree['index_col'], tree['cut_off'])       
        
        if self._isTree(tree['left']): 
            tree['left'] = self.prune(tree['left'], left_subSet)
        
        if self._isTree(tree['right']):
            tree['right'] = self.prune(tree['right'], right_subSet)
        
        if not self._isTree(tree['left']) and not self._isTree(tree['right']):
            left_subSet, right_subSet = self._binSplitDataSet(testData, tree['index_col'], tree['cut_off'])
            errorNoMerge = sum(np.power(left_subSet[:,-1] - tree['left'],2)) + sum(np.power(right_subSet[:,-1] - tree['right'],2))
            treeMean = (tree['left']+tree['right'])/2.0
            errorMerge = sum(np.power(testData[:,-1] - treeMean,2))
            if errorMerge < errorNoMerge:
                print ("merging")
                return treeMean
            else:
                self.trees_ = tree
                return tree
        else:
            self.trees_ = tree
            return tree