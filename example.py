import sys

from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QGraphicsView
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QFileDialog

from graphicsscene import GraphicsScene



class AnnotationWindow(QWidget):
    def __init__(self):
        super(AnnotationWindow,self).__init__()

        self.scene = GraphicsScene()
        self.view = QGraphicsView(self.scene)

        self.button = QPushButton("load image")
        self.button.clicked.connect(self.loadgeofile)

        self.exportbutton = QPushButton("export label")
        self.exportbutton.clicked.connect(self.export)

        layout = QVBoxLayout()
        layout.addWidget(self.button)
        layout.addWidget(self.view)
        layout.addWidget(self.exportbutton)
        self.setLayout(layout)

    def loadgeofile(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file')
        print(fname)
        self.scene.loadgeofile(fname[0],1600,800)

    def export(self):
        fname = QFileDialog.getOpenFileName(self, 'Save to')
        self.scene.export(fname[0])


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = AnnotationWindow()
    widget.resize(640, 480)
    widget.show()
    sys.exit(app.exec_())