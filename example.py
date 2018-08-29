import sys

from PyQt5.QtCore import Qt

from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QGraphicsView
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QFileDialog

from PyQt5.QtCore import QRectF

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvas
import matplotlib.pyplot as plt

from graphicsscene import GraphicsScene
from graphicsview import GraphicsView

import numpy as np

class AnnotationWindow(QWidget):
    def __init__(self):
        super(AnnotationWindow,self).__init__()

        self.scene = GraphicsScene()
        self.view = GraphicsView(self.scene)

        self.button = QPushButton("load image")
        self.button.clicked.connect(self.loadgeofile)

        self.exportbutton = QPushButton("export annotation")
        self.exportbutton.clicked.connect(self.export)

        layout = QVBoxLayout()
        layout.addWidget(self.button)


        sublayout = QHBoxLayout()
        sublayout.addWidget(self.view)

        figure = Figure(figsize=(5, 10))

        subplt = figure.add_subplot(1,4,1)
        subplt.imshow(np.array(range(10)).reshape(10,1), cmap=plt.get_cmap('tab10'), extent=[0,1,0,10], origin='lower')
        tick_gap = (subplt.get_ybound()[1] - subplt.get_ybound()[0])/10
        subplt.set_yticks(np.linspace(subplt.get_ybound()[0]+tick_gap/2, subplt.get_ybound()[1]-tick_gap/2, 10))
        subplt.set_yticklabels([
            '1: Fresh bedrock Proterozoic',
            '2: Moderately weathered bedrock Proterozoic',
            '3: Very highly weathered bedrock Proterozoic',
            '4: Semi-consolidated sediments Cenozoic',
            '5: ',
            '6: ',
            '7: Semi-consolidated sediments Cenozoic',
            '8: Bedrock moderately resistive Palaeozoic',
            '9: Bedrock highly resistive Palaeozoic',
            '10: Very highly weathered bedrock Palaeozoic'
        ])
        subplt.set_xticks([])
        subplt.yaxis.tick_right()


        static_canvas = FigureCanvas(figure)

        sublayout.addWidget(static_canvas)

        layout.addLayout(sublayout)
        layout.addWidget(self.exportbutton)
        self.setLayout(layout)
        self.setWindowTitle("GeoAnnotator")

    def loadgeofile(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file','/g/data1a/ge3/AEM_Model','shapefile (*.shp)')

        if len(fname[0])>0:
            self.scene.loadgeofile(fname[0],1600,800)

    def export(self):
        fname = QFileDialog.getSaveFileName(self, 'Save to')

        if len(fname[0]) > 0:
            self.scene.export(fname[0])


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = AnnotationWindow()
    widget.resize(1024, 768)
    widget.show()
    sys.exit(app.exec_())