import numpy as np

from PyQt5.QtWidgets import QGraphicsView
from PyQt5.QtCore import QTimeLine
from PyQt5.QtCore import Qt

import PyQt5.QtCore as QtCore

class GraphicsView(QGraphicsView):
    def __init__(self, scene):
        super(GraphicsView,self).__init__(scene)
        self.scheduledscaling = 0
        self.drag_switch = False

    #def scale(self, sx, sy):
    #    super(GraphicsView,self).scale(sx, sy)

    def mousePressEvent(self, event):
        super(GraphicsView, self).mousePressEvent(event)
        if event.button() == Qt.RightButton:
            self.drag_switch = True
            self.lastpos=event.pos()
        else:
            self.lastpos=None

    def mouseMoveEvent(self, event):
        super(GraphicsView, self).mouseMoveEvent(event)
        if self.drag_switch == True:
            offset = event.pos()-self.lastpos
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value()+ offset.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() + offset.y())
            self.show()
            self.lastpos=event.pos()

    def mouseReleaseEvent(self, event):
        super(GraphicsView, self).mouseReleaseEvent(event)
        if event.button() == Qt.RightButton:
            self.drag_switch = False


    def wheelEvent(self, event):
        degree = event.angleDelta().y() / 8
        step = degree / 15

        self.scheduledscaling = self.scheduledscaling+step

        animation = QTimeLine(350)
        animation.setUpdateInterval(20)
        animation.valueChanged.connect(animation.currentValue)#Timeline does not start without this line, maybe a QT bug
        animation.valueChanged.connect(lambda:
                                       self.scale(1.0+self.scheduledscaling/300.0, 1.0+self.scheduledscaling/300.0))
        animation.finished.connect(self.reset)
        animation.start()

    def reset(self):
        self.scheduledscaling=0
