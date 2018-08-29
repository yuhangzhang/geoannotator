import numpy as np

from PyQt5.QtWidgets import QGraphicsView
from PyQt5.QtCore import QTimeLine
from PyQt5.QtCore import QObject

import PyQt5.QtCore as QtCore

class GraphicsView(QGraphicsView):
    def __init__(self, scene):
        super(GraphicsView,self).__init__(scene)
        self.scheduledscaling = 0

    def scale1(self, sx, sy):
        print(sx, sy)
        super(GraphicsView,self).scale(sx, sy)


    def wheelEvent(self, event):
        degree = event.angleDelta().y() / 8
        step = degree / 15

        self.scheduledscaling = self.scheduledscaling+step

        animation = QTimeLine(350)
        animation.setUpdateInterval(20)
        animation.valueChanged.connect(lambda: self.scale(1.0+self.scheduledscaling/300.0, 1.0+self.scheduledscaling/300.0))
        animation.finished.connect(self.reset)

        animation.start()

    def reset(self):
        self.scheduledscaling=0
