from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import cv2
from enum import Enum
from PyQt5.QtGui import QImage,QPixmap, QCursor
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

class InteractiveMode(Enum):
    Normal = 0
    Draw = 1

# 图像显示控件
class ImageItem(QGraphicsPixmapItem):
    def __init__(self, parent=None):
        super(ImageItem, self).__init__(parent)
        self.press_sign_ = False
        self.era_mode_ = False
        self.r_ =10
        self.curPix = QPixmap('cursor.png')
        self.scale_rate = 1.
        self.setAcceptHoverEvents(True)

    def hoverEnterEvent(self, event):
        if self.era_mode_ == 0:
            return super(ImageItem, self).hoverEnterEvent(event)
        size = QSize(self.r_*self.scale_rate, self.r_*self.scale_rate)
        pix = self.curPix.scaled(size, Qt.KeepAspectRatio)
        self.my = QCursor(pix)
        QApplication.restoreOverrideCursor()
        QApplication.setOverrideCursor(self.my)

    def hoverLeaveEvent(self, event):
        if self.era_mode_ == 0:
            QApplication.restoreOverrideCursor()
            return super(ImageItem, self).hoverLeaveEvent(event)
        QApplication.restoreOverrideCursor()

    # 鼠标按下事件
    def mousePressEvent(self, event):
        if self.era_mode_ == 0:
            return super(ImageItem, self).mousePressEvent(event)
        else:
            self.press_sign_ = True
            self.__prepoint = event.pos()

    # 鼠标移动事件
    def mouseMoveEvent(self, event):
        if self.era_mode_ == 0:
            return super(ImageItem, self).mousePressEvent(event)
        if self.press_sign_:
            if self.era_mode_==2:
                cv2.line(self.__mask, (int(self.__prepoint.x()), int(self.__prepoint.y())),
                         (int(event.pos().x()), int(event.pos().y())), (0,), self.r_)
            elif self.era_mode_==1:
                cv2.line(self.__mask, (int(self.__prepoint.x()), int(self.__prepoint.y())),
                         (int(event.pos().x()), int(event.pos().y())), (255,), self.r_)
            img = self.__img.copy()
            img[:, :, 0] = cv2.add(self.__img[:, :, 0], self.__mask)
            q_img = QImage(img.data, img.shape[1], img.shape[0], 3 * img.shape[1], QImage.Format_RGB888)
            self.setPixmap(QPixmap.fromImage(q_img))
            self.__prepoint = event.pos()

    # 鼠标移动事件
    def mouseReleaseEvent(self, event):
        if self.era_mode_ == 0:
            return super(ImageItem, self).mousePressEvent(event)
        self.press_sign_ = False

    # 设置当前处理的图像及mask
    def setCurImage(self, img, mask):
        self.__img = img
        self.__mask = mask

    # 设置是否启动擦除模式
    def setEraMode(self, sign):
        self.era_mode_ = sign

    # 设置当前笔刷大小
    def setBrushR(self, r):
        self.r_ = r

class ImageGraphicsView(QGraphicsView):
    def __init__(self,parent):
        super(ImageGraphicsView,self).__init__(parent)
        self.graphScene = QGraphicsScene(self)
        self.setScene(self.graphScene)
        self.setInteractive(True)
        self.setMouseTracking(True)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.scale_rate = 1.
        self.zoom_in = 1.25
        self.interactiveMode = InteractiveMode.Normal
        self.graphItem = ImageItem()
        self.graphItem.setZValue(0)
        self.graphScene.addItem(self.graphItem)

    def wheelEvent(self, event):
        delta = event.angleDelta()
        if delta.y() > 0:
            self.scale_rate = self.zoom_in
        else:
            self.scale_rate = 1 / self.zoom_in
        # if event.modifiers() & QtCore.Qt.ShiftModifier:
        # else:# event.modifiers():
        self.graphItem.scale_rate *= self.scale_rate
        self.graphItem.hoverEnterEvent(None)
        # Set Anchors
        self.setResizeAnchor(QtWidgets.QGraphicsView.NoAnchor)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.NoAnchor)

        cur_pos = self.mapToScene(event.pos())
        self.scale(self.scale_rate, self.scale_rate)
        new_pos = self.mapToScene(event.pos())
        delta_zoomed = new_pos - cur_pos
        self.translate(delta_zoomed.x(), delta_zoomed.y())
        event.accept()
        self.graphScene.update()
        # return
        return super(ImageGraphicsView, self).wheelEvent(event)

    def mousePressEvent(self,event: QtWidgets.QGraphicsSceneMouseEvent):
        return super(ImageGraphicsView,self).mousePressEvent(event)

    def mouseMoveEvent(self, event:QtWidgets.QGraphicsSceneMouseEvent):
        return super(ImageGraphicsView,self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        return super(ImageGraphicsView,self).mouseReleaseEvent(event)

    def SetInteractiveMode(self,mode):
         if mode == 0: #normal
             self.interactiveMode = InteractiveMode.Normal
             self.setDragMode(QGraphicsView.ScrollHandDrag)
             self.graphItem.setSelected(False)
             self.graphItem.setEraMode(0)
             self.graphItem.hoverEnterEvent(None)
             self.graphItem.setFlag(~QGraphicsItem.ItemIsMovable & ~QGraphicsItem.ItemIsSelectable)

         elif mode == 1 or mode == 2:#Draw and erase
             self.interactiveMode = InteractiveMode.Draw
             self.setDragMode(QGraphicsView.NoDrag)
             self.graphItem.setFlag(QGraphicsItem.ItemIsMovable & QGraphicsItem.ItemIsSelectable)
             self.graphItem.setEraMode(mode)
             self.graphItem.hoverEnterEvent(None)

