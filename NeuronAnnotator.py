import os
import sys
import tifffile
import cv2
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog,QVBoxLayout,QMessageBox
from PyQt5.QtGui import QImage, QPixmap,QIcon
from ImageAnnotationDock import ImageAnnotationDock
from ImageGraphicsView import ImageGraphicsView

from GDataManager import GDataManager

class NeuronAnnotator(QMainWindow):
    def __init__(self, parent=None):
        super(NeuronAnnotator, self).__init__(parent)
        self.setObjectName('NeuronAnnotator')
        self.resize(1024,768)
        self.dataManager = GDataManager()
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate('NeuronAnnotator','CellSuperSegmentor'))
        self.setWindowIcon(QIcon('./title.png'))
        self.CreateLeftDock()
        self.SetupUI()

        self.dataManager.images = None
        self.dataManager.mask = None
        # 信号槽连接
        self.singleDock.open_pb.clicked.connect(self.openFileSlot)
        self.singleDock.open_label.clicked.connect(self.openLabelSlot)
        self.singleDock.save_pb.clicked.connect(self.saveFileSlot)
        self.singleDock.split_orig.clicked.connect(self.splitOrigSlot)
        self.singleDock.create_orig.clicked.connect(self.Change_orig)
        self.singleDock.create_bin.clicked.connect(self.Change_bin)
        self.singleDock.view_cbox.currentIndexChanged.connect(self.viewChangeSlot)
        self.singleDock.frame_sbox.valueChanged.connect(self.frameChangeSlot)
        self.singleDock.mode_cbox.currentIndexChanged.connect(self.modeChangeSlot)
        self.singleDock.r_sbox.valueChanged.connect(self.graphView.graphItem.setBrushR)

    def keyPressEvent(self,event):
        if event.key() == Qt.Key_1:
            self.singleDock.mode_cbox.setCurrentIndex(0)
            #self.graphView.SetInteractiveMode(0)
        elif event.key() == Qt.Key_2:
            self.singleDock.mode_cbox.setCurrentIndex(1)
        elif event.key() == Qt.Key_3:
            self.singleDock.mode_cbox.setCurrentIndex(2)

    def CreateLeftDock(self):
        self.singleDock = ImageAnnotationDock(self)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.singleDock)
        self.singleDock.SetDataManager(self.dataManager)
        self.singleDock.show()

    def SetupUI(self):
        self.mainWidget = QtWidgets.QWidget(self)
        self.centralLayout = QVBoxLayout(self.mainWidget)
        self.setCentralWidget(self.mainWidget)
        self.mainWidget.setLayout(self.centralLayout)
        self.graphView = ImageGraphicsView(self.mainWidget)
        self.centralLayout.addWidget(self.graphView)

    def Change_orig(self):
        # 文件对话框
        file_name = QFileDialog.getOpenFileNames(self, 'open file', './', '3D image(*.tif)')
        #print(len(file_name[0]))
        volume = []
        Files_Path = './orig'  # 把你的所有图片放在py文件同目录下的volume文件夹
        Res_path = './create_orig'
        #imgList = os.listdir(Files_Path)  # 读取文件目录下的所有文件名
        #imgList = file_name

        #imgList.sort(key=lambda x: int(x.split('.')[0]))  # 按照数字顺序排列图片，图片名示例：1.tif, 2.tif, 3.tif, 4.tif, 5.tif ...
        for count in range(0, len(file_name[0])):
            # tif = TIFF.open(Files_Path + '/' + imgList[count], mode='r')
            tif = tifffile.imread(file_name[0][count])
            tif = np.array(tif)
            if tif.shape[0] == 1:
                tif = tif.squeeze(0)
            # else:
            #     assert tif.shape[0] == 1
            volume.append(tif)
        volume = np.array(volume)
        print('Read success.')
        tifffile.imsave(os.path.join(Res_path, 'res.tif'), volume)
        # tifffile.imsave(Files_Path + '.tif', volume)

    def Change_bin(self):
        # 文件对话框
        file_name = QFileDialog.getOpenFileNames(self, 'open file', './', '3D image(*.tif)')
        # print(len(file_name[0]))
        volume = []
        Files_Path = './mask-manual-bin'  # 把你的所有图片放在py文件同目录下的volume文件夹
        Res_path = './create_bin'
        # imgList = os.listdir(Files_Path)  # 读取文件目录下的所有文件名
        # imgList = file_name

        # imgList.sort(key=lambda x: int(x.split('.')[0]))  # 按照数字顺序排列图片，图片名示例：1.tif, 2.tif, 3.tif, 4.tif, 5.tif ...
        for count in range(0, len(file_name[0])):
            # tif = TIFF.open(Files_Path + '/' + imgList[count], mode='r')
            tif = tifffile.imread(file_name[0][count])
            tif = np.array(tif)
            if tif.shape[0] == 1:
                tif = tif.squeeze(0)
            # else:
            #     assert tif.shape[0] == 1
            volume.append(tif)
        volume = np.array(volume)
        print('Read success.')
        tifffile.imsave(os.path.join(Res_path, 'res.tif'), volume)
        # tifffile.imsave(Files_Path + '.tif', volume)

    # 打开文件槽函数
    def openFileSlot(self):
        # 文件对话框
        file_name=QFileDialog.getOpenFileName(self, 'open file','./','3D image(*.tif)')
        if(file_name[0]==''):
            return
        else:
            # 读取tif文件
            self.dataManager.images = tifffile.imread(file_name[0])
            if self.dataManager.images.dtype == np.uint16:
                self.dataManager.displayImages = self.dataManager.images / (np.max(self.dataManager.images)) * 255
                self.dataManager.displayImages = self.dataManager.displayImages.astype(np.uint8)
            else:
                self.dataManager.displayImages = self.dataManager.images
            self.dataManager.mask = np.zeros(self.dataManager.images.shape,
                                             dtype=np.uint8)
            self.cur_view_idx = 0
            self.cur_frame_idx = 0

            img = self.dataManager.displayImages[0,:,:]
            mask = self.dataManager.mask[0,:,:]

            # 灰度单通道转RGB三通道图像
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

            # 记录当前正在处理的图像
            self.dataManager.cur_img = img.copy()
            self.dataManager.cur_mask = mask.copy()

            # 将图像信息写入qlabel对象内
            self.graphView.graphItem.setCurImage(self.dataManager.cur_img,self.dataManager.cur_mask)

            # 原图+标注mask
            img[:,:,0] = cv2.add(img[:,:,0],mask)

            # 转化成QImage
            q_img = QImage(img.data, img.shape[1], img.shape[0], 3*img.shape[1], QImage.Format_RGB888)

            # 显示图像
            self.graphView.graphItem.setPixmap(QPixmap.fromImage(q_img))

    # 打开标注文件槽函数
    def openLabelSlot(self):
        # if self.images is None:
        #     QMessageBox.warning(self,'warning', 'please open original image first')
        #     return
        # 文件对话框
        dirName = QFileDialog.getExistingDirectory(self, 'open label', './')
        dirList = os.listdir(dirName)
        self.mask = np.zeros(self.images.shape, dtype = self.images.dtype)
        for k in range(len(dirList)):
            # 读取tif文件
            self.dataManager.mask[k,:] = tifffile.imread(os.path.join(dirName, dirList[k]))

        self.cur_view_idx = 0
        self.cur_frame_idx = 0

        img = self.dataManager.images[0, :, :]
        mask = self.dataManager.mask[0, :, :]

        # 灰度单通道转RGB三通道图像
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # 记录当前正在处理的图像
        self.dataManager.cur_img = img.copy()
        self.dataManager.cur_mask = mask.copy()

        # 将图像信息写入qlabel对象内
        self.graphView.graphItem.setCurImage(self.dataManager.cur_img, self.dataManager.cur_mask)

        # 原图+标注mask
        img[:, :, 0] = cv2.add(img[:, :, 0], mask)

        # 转化成QImage
        q_img = QImage(img.data, img.shape[1], img.shape[0], 3 * img.shape[1], QImage.Format_RGB888)

        # 显示图像
        #self.resize(img.shape[1] + 178, img.shape[0])
        self.graphView.graphItem.setPixmap(QPixmap.fromImage(q_img))

        self.singleDock.frame_sbox.setValue(0)


    # 视角切换槽函数
    def viewChangeSlot(self,view_idx):
        self.singleDock.frame_sbox.setValue(0)
        self.showImageChange(view_idx,0)

    # 当前帧切换槽函数
    def frameChangeSlot(self,frame_idx):
        view_idx=self.singleDock.view_cbox.currentIndex()
        self.showImageChange(view_idx,frame_idx)

    def writeMask(self):
        if (self.cur_view_idx==0):
            self.dataManager.mask[self.cur_frame_idx, :, :]=self.dataManager.cur_mask.copy()
        elif(self.cur_view_idx==1):
            self.dataManager.mask[:, self.cur_frame_idx, :]=self.dataManager.cur_mask.copy()
        else:
            self.dataManager.mask[:,:, self.cur_frame_idx] = self.dataManager.cur_mask.copy()


    # 更新窗口显示的图像
    def showImageChange(self,view_idx,frame_idx):
        self.writeMask()
        if (view_idx==0):
            self.singleDock.frame_sbox.setMaximum(self.dataManager.images.shape[0]-1)
            img = self.dataManager.displayImages[frame_idx, :, :]
            mask = self.dataManager.mask[frame_idx, :, :]
        elif(view_idx==1):
            self.singleDock.frame_sbox.setMaximum(self.dataManager.images.shape[1]-1)
            img = self.dataManager.displayImages[:, frame_idx, :]
            mask = self.dataManager.mask[:, frame_idx, :]
        else:
            self.singleDock.frame_sbox.setMaximum(self.dataManager.images.shape[2]-1)
            img = self.dataManager.displayImages[:, :, frame_idx]
            mask = self.dataManager.mask[:, :, frame_idx]

        self.cur_view_idx=view_idx
        self.cur_frame_idx=frame_idx
        img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        self.dataManager.cur_img=img.copy()
        self.dataManager.cur_mask=mask.copy()
        self.graphView.graphItem.setCurImage(self.dataManager.cur_img, self.dataManager.cur_mask)

        img[:, :, 0] = cv2.add(img[:, :, 0], mask)
        q_img = QImage(img.data, img.shape[1], img.shape[0], 3 * img.shape[1], QImage.Format_RGB888)
        self.graphView.graphItem.setPixmap(QPixmap.fromImage(q_img))

        #重置显示图像的窗口大小
        #self.resize(img.shape[1] + 178, img.shape[0])

    # def closeEvent(self, event):
    #     pass

    def splitOrigSlot(self):
        file_name = QFileDialog.getSaveFileName(self, 'split original image', './', '3D image(*.tif)')
        if (file_name[0] == ''):
            return
        else:
            sz = self.images.shape[0]
            for i in range(sz):
                saveName = file_name[0][:-4] + '_%05d.tif' % i
                tifffile.imwrite(saveName, self.dataManager.images[i, :])

    def saveFileSlot(self):
       file_name = QFileDialog.getSaveFileName(self,'save label','./','3D image(*.tif)')
       if (file_name[0] == ''):
           return
       else:
            #写入tif文件
            self.writeMask()
            sz = self.dataManager.mask.shape[0]
            for i in range(sz):
                saveName = file_name[0][:-4]+'_%05d.tif'%i
                tifffile.imwrite(saveName, self.dataManager.mask[i,:])


    #标注及擦除模式切换
    def modeChangeSlot(self,idx):
        self.graphView.SetInteractiveMode(idx)
