import numpy as np
from sklearn import neighbors
from metric_learn import LMNN



class GeoLMNN(neighbors.KNeighborsClassifier):
    def __init__(self, n_neighbors=3):
        super(GeoLMNN, self).__init__(n_neighbors=n_neighbors)
        self.lmnn = LMNN(n_neighbors)



    def fit(self, X, y):
        self.lmnn.fit(X, y)
        super(GeoLMNN, self).fit(self.lmnn.transform(X), y)


    def predict(self, X):
        y = super(GeoLMNN, self).predict(self.lmnn.transform(X))
        return y
