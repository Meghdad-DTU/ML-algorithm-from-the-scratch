import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

class AdaBoost(object):
    """ AdaBoost enemble classifier from scratch """
    def __init__(self, classifier = "DecisionTree"):
        self.classifier = classifier
        self.stumps = None
        self.stump_weights = None
        self.errors = None
        self.weights = None
        self.X = None
        self.y = None

    def _check_X_y(self, X, y):
        """ Validate assumptions about format of input data"""
        assert type(X) and type(y) == np.ndarray, 'Expecting the type of input to be array'        
        assert list(range(len(set(y)))) == list(set(y)), 'Expecting response labels to be 0,1,2,...'
        return X, y 
    
    def _bootstrap_sampling(self, X, y, n, weights):
        """resample n number of data"""
        selectedIndices = np.random.choice(range(X.shape[0]), size=n, replace=True, p=weights)        
        y = y[selectedIndices]
        X = X[selectedIndices,:]
        return (X,y)
    
    def fit(self, X: np.ndarray, y: np.ndarray, maxIter: int):
        """ Fit the model using training data """
        X, y = self._check_X_y(X, y)
        self.X = X
        self.y = y
        n = X.shape[0]

        # init numpy arrays
        self.weights = np.zeros((maxIter, n))
        self.stumps = np.zeros(maxIter, dtype=object)
        self.stump_weights = np.zeros(maxIter)
        self.errors = np.zeros(maxIter)

        # initialize weights uniformly
        self.weights[0] = np.ones(n) / n

        cls = [DecisionTreeClassifier(),
               LogisticRegression(solver = "lbfgs", multi_class="multinomial", max_iter=1000),
               KNeighborsClassifier(n_neighbors=10)]
        
        for t in range(maxIter):
            # fit  weak learner
            weights = self.weights[t]
            if self.classifier == "DecisionTree":
                stump = cls[0]  
            elif self.classifier == "LogisticRegression":
                stump = cls[1]
            elif self.classifier == "KNeighbors":
                stump = cls[2]
            else:
                raise Exception("Unknown classifier: %s. Valids are: DecisionTree, LogisticRegression & KNeighbors." % (classifier))
                
            while True : 
                # not a thing of beauty, however log.reg. fails if presented with less than two classes. 
                X, y = self._bootstrap_sampling(self.X, self.y, n, weights=weights)
                uniqVal = list(set(y))
                if not (all(y == uniqVal[0]) or all(y == uniqVal[1])) : break  
            
            stump = stump.fit(X, y)
            # calculate error and stump weight from weak learner prediction
            stump_pred = stump.predict(X)
            err = weights[(stump_pred != y)].sum()# / n
                      
            # adding a small epsilon to the numerator and denominator for stability
            epsilon = 1e-5
            stump_weight = 0.5 * np.log((1 - err + epsilon) / (err + epsilon)) 

            # update sample weights
            index = np.where(stump_pred != y)[0]
            weights[index] = weights[index]*np.exp(-stump_weight) # increase weights of misclassified observations
            weights = weights / np.sum(weights)               # renormalize weights

            # If not final iteration, update sample weights for t+1
            if t+1 < maxIter:
                self.weights[t+1] = weights

            # save results of iteration
            self.stumps[t] = stump
            self.stump_weights[t] = stump_weight
            self.errors[t] = err

        return self
    
    def score(self, X=None ):
        if X is not None:
            self.X = X
        stump_preds = np.array([stump.predict(self.X) for stump in self.stumps])
        
        ## method 1:
        arr = np.dot(self.stump_weights, stump_preds)
        ## method 2:
        #arr = stump_preds.mean(axis=0)
        
        cutOff = (max(arr))/ len(set(self.y)) ## unique response lable          
        ## replace value within range
        for n in range(len(set(self.y))): 
            arr[(arr > n*cutOff) & (arr <= (n+1)*cutOff)] = n
        self.predicted = arr
        return [pred == true for pred,true in zip(arr, self.y)].count(True)/len(self.y)
    
    def prediction(self, X):
        """ Make predictions using already fitted model """
        self.score(X)
        return self.predicted