import numpy as np
import shapefile as sf
import sklearn.decomposition as sd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

from geoinput import GeoInput

class GeoFile(GeoInput):
    def __init__(self, filename):

        if filename.endswith('shp'):
            inputfile = sf.Reader(filename)
            inputrecord = np.array(inputfile.records())

            super(GeoFile, self).__init__(inputrecord)

