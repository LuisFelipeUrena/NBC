import numpy as np
import scipy as scy

class NaiveBayesClassifier:
    '''

    '''
    def fit(self,x,y):
        '''
        input: 
            x: array-like, features to be used in the model
            y: target variable
        returns:





        '''
        # get the prior probability
        # get the mean and variance of each class
        n_samples, n_features = x.shape
        self._classes = np.unique(y) # identifies unique classes
        n_classes = len(self._classes) # number of classes
        
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64) # mean
        self._var = np.zeros((n_classes, n_features), dtype=np.float64) #variance
        self._prior = np.zeros(n_classes,dtype=np.float64) #prior probabilities

        # fill out the mean, variance and prior probabilities zero arrays with the correct data
        for i, c in enumerate(self._classes):
            x_classes = x[y==c]
            self._mean[i,:] = x_classes.mean(axis=0)
            self._var[i,:] = x_classes.var(axis=0)
            self._prior[i] = x_classes.shape[0] / float(n_samples)

    def predict(self,x):
        y_pred = [self._predict(i) for i in x]
        return np.array(y_pred)
    
    def _predict(self, x):
        posterior = []

        for i, c in enumerate(self._classes):
            prior = np.log(self._prior[i])
            class_cond = np.sum(np.log(self._pdf(i, x)))
            post = prior + class_cond
            posterior.append(post)

        return self._classes[np.argmax(posterior)]

    def _pdf(self,class_ids,x):
        mean = self._mean[class_ids]
        var = self._var[class_ids]
        
        numerator = np.exp(- (x-mean)**2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator