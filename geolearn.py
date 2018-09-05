

from itertools import cycle, compress
from sklearn import preprocessing
from sklearn import neighbors
from metric_learn import LMNN
from multiprocessing import Pool

class geolearn:
    def __init__(self, model):
        self.model = model


    def fit(self,feature,label):
        model.fit(feature, label)

    def predict(self, feature):
        return model.predict(feature)
