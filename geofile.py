import numpy as np
import shapefile as sf
import sklearn.decomposition as sd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

class GeoFile:
    def __init__(self, filename):

        if filename.endswith('shp'):
            inputfile = sf.Reader(filename)
            inputrecord = np.array(inputfile.records())

            super(GeoFile, self).__init__(inputrecord)

