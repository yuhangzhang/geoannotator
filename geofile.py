import numpy as np
import shapefile as sf
import sklearn.decomposition as sd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

from geodata import GeoData

class GeoFile(GeoData):
    def __init__(self, filename):

        if filename.endswith('shp'):
            inputfile = sf.Reader(filename)
            inputrecord = np.array(inputfile.records()[1:10000])

            super(GeoFile, self).__init__(inputrecord)

