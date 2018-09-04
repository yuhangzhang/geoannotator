import numpy as np

from skimage.draw import polygon

import matplotlib.pyplot as plt

from PyQt5.QtWidgets import QGraphicsScene

from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QPolygon
from PyQt5.QtGui import QPolygonF
from PyQt5.QtGui import QBrush
from PyQt5.QtGui import QPainterPath
from PyQt5.QtGui import QPen
from PyQt5.QtGui import QColor

from PyQt5.QtCore import Qt

from geofile import GeoFile
from geodatabase import GeoDataBase
from dialog import Dialog
from dialogdropdown import DialogDropDown

class GraphicsScene(QGraphicsScene):
    def __init__(self):
        super(GraphicsScene, self).__init__()
        self.draw_switch = False
        #self.dialog = Dialog()
        self.dialog = DialogDropDown()
        self.pixmap = QPixmap()
        self.pixmaptopdown = QPixmap()
        self.label_all = np.zeros([0,0])
        self.pixmaphandle = None

    def loaddatabase(self, width, height):
        self.geofile = GeoDataBase()
        arr = self.geofile.getimagexy(width, int(height/4))
        qimg = QImage(arr, arr.shape[1], arr.shape[0], QImage.Format_RGB888)#.rgbSwapped()
        self.pixmaptopdown = QPixmap(qimg)
        self.pixmaptopdownhandle = self.addPixmap(self.pixmaptopdown)

        arr = self.geofile.getimage(width, height)
        qimg = QImage(arr, arr.shape[1], arr.shape[0], QImage.Format_RGB888)#.rgbSwapped()
        self.pixmap = QPixmap(qimg)
        self.pixmaphandle = self.addPixmap(self.pixmap)
        self.pixmaptopdownhandle.moveBy(0, -self.pixmaptopdown.height()-20)

        self.label_all = np.zeros([self.pixmap.height(), self.pixmap.width()])  #initiate label image

    def openfile(self, filename, width, height):
        self.geofile = GeoFile(filename)
        arr = self.geofile.getimage(width, height)
        qimg = QImage(arr, arr.shape[1], arr.shape[0], QImage.Format_RGB888).rgbSwapped()
        self.pixmap = QPixmap(qimg)
        self.pixmaphandle = self.addPixmap(self.pixmap)
        self.label_all = np.zeros([self.pixmap.height(), self.pixmap.width()])  #initiate label image


    def mousePressEvent(self, event):
        super(GraphicsScene, self).mousePressEvent(event)

        # prepare for a new crop
        if event.button() == Qt.LeftButton:
            self.draw_switch = True

            print(event.scenePos().x(), event.scenePos().y())

            if event.scenePos().x()>=0 \
                    and event.scenePos().x()<self.pixmap.width() \
                    and event.scenePos().y()>=0 \
                    and event.scenePos().y()<self.pixmap.height():
                self.lastpos = event.pos()
                self.poly = [self.lastpos]
            else:
                self.lastpos = None
                self.poly = []

            self.pathset = []

    def mouseMoveEvent(self, event):
        super(GraphicsScene, self).mousePressEvent(event)

        pos = event.scenePos()

        if self.draw_switch == True \
                and pos.x()>=0 \
                and pos.x()<self.pixmap.width() \
                and pos.y()>=0 \
                and pos.y()<self.pixmap.height():
            if self.lastpos is not None:
                # show trace on the screen
                path = QPainterPath()
                path.setFillRule(Qt.WindingFill)
                path.moveTo(self.lastpos)
                path.lineTo(pos)
                self.pathset.append(self.addPath(path, pen=QPen(Qt.white)))
                self.poly.append(pos)  # keep vertex for label generation later

            self.lastpos = pos  # update

    def mouseReleaseEvent(self, event):
        super(GraphicsScene, self).mousePressEvent(event)

        pos = event.scenePos()

        if event.button() == Qt.LeftButton:
            self.draw_switch = False

            if pos.x()>=0 \
            and pos.x()<self.pixmap.width() \
            and pos.y()>=0 \
            and pos.y()<self.pixmap.height():
                label = self.dialog.gettext()   #ask for label

                if label[1]==True and len(label[0])>0 and len(self.poly)>0:  #if user input a label


                    # save the label on backend
                    if len(self.poly)>1:
                        # point the label on screen
                        poly = QPolygon()
                        for p in self.poly:
                            poly.append(p.toPoint())
                            # print(p)
                        brush = QBrush()
                        labelcolor = QColor(*[c*255 for c in plt.get_cmap('tab10').colors[int(label[0])-1]])
                        brush.setColor(labelcolor)
                        brush.setStyle(Qt.SolidPattern)
                        self.addPolygon(QPolygonF(poly), pen=QPen(labelcolor), brush=brush)
                        x, y = polygon([p.toPoint().x() for p in self.poly],[p.toPoint().y() for p in self.poly])
                    else:
                        self.addEllipse(self.poly[0].x(),self.poly[0].y(),2,2,pen=QPen(Qt.red))
                        x = self.poly[0].toPoint().x()
                        y = self.poly[0].toPoint().y()
                    self.label_all[y, x] = int(label[0])

            # remove the trace painted so far
            for p in self.pathset:
                self.removeItem(p)

    def export(self, filename):
        fw = open(filename, 'w')

        for h in range(self.label_all.shape[0]):
            for w in range(self.label_all.shape[1]):
                if self.label_all[h,w]>0:
                    points = self.geofile.getpoint(w, h)
                    for p in points:
                        np.savetxt(fw, np.append(p[0:-2],self.label_all[h,w]).reshape(1, -1), fmt='%s')

        fw.close()


    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            if self.pixmaphandle is not None:
                self.pixmaphandle.setZValue(1)


    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space:
            if self.pixmaphandle is not None:
                self.pixmaphandle.setZValue(-1)