import sys

from PyQt5.QtCore import Qt

from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QGraphicsView
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QFileDialog

from PyQt5.QtCore import QRectF

from graphicsscene import GraphicsScene
from graphicsview import GraphicsView



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
        layout.addWidget(self.view)
        layout.addWidget(self.exportbutton)
        self.setLayout(layout)
        self.setWindowTitle("GeoAnnotator")

    def loadgeofile(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file','/g/data1a/ge3/AEM_Model','shapefile (*.shp)')
        print(fname)

        if len(fname[0])>0:
            self.scene.loadgeofile(fname[0],1600,800)



    def export(self):
        fname = QFileDialog.getSaveFileName(self, 'Save to')

        if len(fname[0]) > 0:
            self.scene.export(fname[0])


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = AnnotationWindow()
    widget.resize(640, 480)
    widget.show()
    sys.exit(app.exec_())