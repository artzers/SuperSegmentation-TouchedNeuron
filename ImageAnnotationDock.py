# -*- coding: utf-8 -*-

import os
import torch.cuda
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.Qt import QThread
from PyQt5.QtWidgets import QMainWindow, QAction, qApp, QApplication, \
QStatusBar, QMenuBar, QFileDialog, QMessageBox, QSpacerItem,\
    QDoubleSpinBox, QGroupBox, QPushButton,\
    QDockWidget, QRadioButton, QButtonGroup

from Segmentor import Segmentor


class TrainThread(QThread):
    def __init__(self):
        super().__init__()
        self.segmentor = None
        self.env = None
    def SetWorker(self, arg):
        self.segmentor = arg
    def SetEnv(self, arg):
        self.env = arg
    def run(self):
        retVal = self.segmentor.Train(self.env.LRSampleEdit.text(),
                                      self.env.HRSampleEdit.text(),
                                      self.env.meanDoubleSpinBox1.value(),
                                      self.env.stdDoubleSpinBox1.value(),
                                      self.env.epochSpinBox.value(),
                                      self.env.visCheckBox.isChecked(),
                                      initLR = self.env.learnRateSpinBox.value(),
                                      decayTurn = self.env.decaySpinBox.value(),
                                      savePath = self.env.trainPathEdit.text())

    def stop(self):
        if self.isRunning():
            self.terminate()

class ImageAnnotationDock(QtWidgets.QDockWidget):
    def __init__(self, parent=None):
        super(ImageAnnotationDock, self).__init__(parent)

        self.setObjectName('ImageAnnotationDock')
        self.CreateDockWidget()
        self.segmentor = Segmentor()
        self.trainFlag = False
        self.thread = None

    def SetDataManager(self, manager):
        self.dataManager = manager

    def CreateDockWidget(self):
        self.dockWidget = QtWidgets.QWidget(self)
        self.setWidget(self.dockWidget)
        self.verticLayout = QtWidgets.QVBoxLayout(self.dockWidget)

        self.imageGroupBox = QtWidgets.QGroupBox(self.dockWidget)
        self.imageGroupBox.setObjectName("imageGroupBox")
        self.imageGridLayout = QtWidgets.QGridLayout(self.imageGroupBox)
        self.imageGridLayout.setObjectName("imageGridLayout")
        self.imageGroupBox.setLayout(self.imageGridLayout)
        grid = self.imageGroupBox.layout()
        self.label_1 = QtWidgets.QLabel(self.dockWidget)
        self.label_1.setObjectName("label_1")

        self.view_cbox = QtWidgets.QComboBox(self.dockWidget)
        self.view_cbox.setObjectName("view_cbox")
        self.view_cbox.addItem("")
        self.view_cbox.addItem("")
        self.view_cbox.addItem("")

        self.label_2 = QtWidgets.QLabel(self.dockWidget)
        self.label_2.setObjectName("label_2")

        self.frame_sbox = QtWidgets.QSpinBox(self.dockWidget)
        self.frame_sbox.setMaximum(10000)
        self.frame_sbox.setObjectName("frame_sbox")

        self.label_3 = QtWidgets.QLabel(self.dockWidget)
        self.label_3.setObjectName("label_3")

        self.mode_cbox = QtWidgets.QComboBox(self.dockWidget)
        self.mode_cbox.setObjectName("mode_cbox")
        self.mode_cbox.addItem("")
        self.mode_cbox.addItem("")
        self.mode_cbox.addItem("")


        self.label_4 = QtWidgets.QLabel(self.dockWidget)
        self.label_4.setObjectName("label_4")

        self.r_sbox = QtWidgets.QSpinBox(self.dockWidget)
        self.r_sbox.setMaximum(50)
        self.r_sbox.setProperty("value", 10)
        self.r_sbox.setObjectName("r_sbox")


        self.open_pb = QtWidgets.QPushButton(self.dockWidget)
        self.open_pb.setObjectName("open_pb")

        self.open_label = QtWidgets.QPushButton(self.dockWidget)
        self.open_label.setObjectName("save_label")

        self.split_orig = QtWidgets.QPushButton(self.dockWidget)
        self.split_orig.setObjectName("split_orig")

        self.save_pb = QtWidgets.QPushButton(self.dockWidget)
        self.save_pb.setObjectName("save_pb")

        self.create_orig = QtWidgets.QPushButton(self.dockWidget)
        self.create_orig.setObjectName("create_orig")

        self.create_bin = QtWidgets.QPushButton(self.dockWidget)
        self.create_bin.setObjectName("create_bin")

        grid.addWidget(self.label_1, 0, 0, 1, 1)
        grid.addWidget(self.view_cbox, 0, 1, 1, 1)
        grid.addWidget(self.label_2, 1, 0, 1, 1)
        grid.addWidget(self.frame_sbox, 1, 1, 1, 1)
        grid.addWidget(self.label_3, 2, 0, 1, 1)
        grid.addWidget(self.mode_cbox, 2, 1, 1, 1)
        grid.addWidget(self.label_4, 3, 0, 1, 1)
        grid.addWidget(self.r_sbox, 3, 1, 1, 1)
        grid.addWidget(self.open_pb, 4, 0, 1, 1)
        grid.addWidget(self.open_label, 4, 1, 1, 1)
        grid.addWidget(self.split_orig, 5, 0, 1, 1)
        grid.addWidget(self.save_pb, 5, 1, 1, 1)
        grid.addWidget(self.create_orig, 6, 0, 1, 1)
        grid.addWidget(self.create_bin, 6, 1, 1, 1)


        self.verticLayout.addWidget(self.imageGroupBox)

        # train setting
        self.groupBox1 = QGroupBox('Training')
        self.verticLayout.addWidget(self.groupBox1)
        gridLayout = QtWidgets.QGridLayout(self.groupBox1)
        self.groupBox1.setLayout(gridLayout)
        grid = self.groupBox1.layout()
        meanLabel1 = QtWidgets.QLabel(self.dockWidget)
        meanLabel1.setText('Mean Val')
        self.meanDoubleSpinBox1 = QtWidgets.QDoubleSpinBox(self.dockWidget)
        self.meanDoubleSpinBox1.setValue(35.)
        grid.addWidget(meanLabel1, 0, 0, 1, 1)
        grid.addWidget(self.meanDoubleSpinBox1, 0, 1, 1, 1)
        #
        stdLabel1 = QtWidgets.QLabel(self.dockWidget)
        stdLabel1.setText('Std Val')
        self.stdDoubleSpinBox1 = QtWidgets.QDoubleSpinBox(self.dockWidget)
        self.stdDoubleSpinBox1.setValue(35.)
        grid.addWidget(stdLabel1, 1, 0, 1, 1)
        grid.addWidget(self.stdDoubleSpinBox1, 1, 1, 1, 1)
        # sample path
        self.LRSampleEdit = QtWidgets.QLineEdit(self.dockWidget)
        self.LRSampleButton = QtWidgets.QPushButton(self.dockWidget)
        self.LRSampleButton.clicked.connect(self.SetLRSamplePath)
        self.LRSampleButton.setText('Orig Path')
        self.HRSampleEdit = QtWidgets.QLineEdit(self.dockWidget)
        self.HRSampleButton = QtWidgets.QPushButton(self.dockWidget)
        self.HRSampleButton.clicked.connect(self.SetHRSamplePath)
        self.HRSampleButton.setText('Label Path')
        grid.addWidget(self.LRSampleEdit, 2, 0, 1, 1)
        grid.addWidget(self.LRSampleButton, 2, 1, 1, 1)
        grid.addWidget(self.HRSampleEdit, 3, 0, 1, 1)
        grid.addWidget(self.HRSampleButton, 3, 1, 1, 1)
        # save path
        self.trainPathEdit = QtWidgets.QLineEdit(self.dockWidget)
        self.trainPathEdit.setText(os.getcwd() + '/')
        self.trainPathButton = QtWidgets.QPushButton(self.dockWidget)
        self.trainPathButton.clicked.connect(self.SetTrainSavePath)
        self.trainPathButton.setText('Set Save Path')
        grid.addWidget(self.trainPathEdit, 4, 0, 1, 1)
        grid.addWidget(self.trainPathButton, 4, 1, 1, 1)
        epochLabel = QtWidgets.QLabel(self.dockWidget)
        epochLabel.setText('epoch num')
        self.epochSpinBox = QtWidgets.QSpinBox(self.dockWidget)
        self.epochSpinBox.setMaximum(100000)
        self.epochSpinBox.setValue(500)
        grid.addWidget(epochLabel, 5, 0, 1, 1)
        grid.addWidget(self.epochSpinBox, 5, 1, 1, 1)
        #
        learingRateLabel = QtWidgets.QLabel(self.dockWidget)
        learingRateLabel.setText('LearningRate')
        self.learnRateSpinBox = QtWidgets.QDoubleSpinBox(self.dockWidget)
        self.learnRateSpinBox.setDecimals(7)
        self.learnRateSpinBox.setValue(0.002)
        grid.addWidget(learingRateLabel, 6, 0, 1, 1)
        grid.addWidget(self.learnRateSpinBox, 6, 1, 1, 1)
        #
        decayTurnLabel = QtWidgets.QLabel(self.dockWidget)
        decayTurnLabel.setText('Decay Turn')
        self.decaySpinBox = QtWidgets.QSpinBox(self.dockWidget)
        self.decaySpinBox.setMaximum(99999)
        self.decaySpinBox.setValue(2000)
        grid.addWidget(decayTurnLabel, 7, 0, 1, 1)
        grid.addWidget(self.decaySpinBox, 7, 1, 1, 1)
        #
        self.visCheckBox = QtWidgets.QCheckBox(self.dockWidget)
        self.visCheckBox.setText('enable visdom')
        self.visCheckBox.setChecked(False)
        grid.addWidget(self.visCheckBox, 8, 0, 1, 1)
        self.trainButton = QPushButton(self.dockWidget)
        self.trainButton.setText('Train')
        self.trainButton.clicked.connect(self.Train)
        grid.addWidget(self.trainButton, 8, 1, 1, 1)
        # prediction setting
        self.groupBox2 = QGroupBox('Prediction')
        self.verticLayout.addWidget(self.groupBox2)
        gridLayout = QtWidgets.QGridLayout(self.groupBox2)
        self.groupBox2.setLayout(gridLayout)
        grid = self.groupBox2.layout()
        meanLabel2 = QtWidgets.QLabel(self.dockWidget)
        meanLabel2.setText('Mean Val')
        self.meanDoubleSpinBox2 = QtWidgets.QDoubleSpinBox(self.dockWidget)
        self.meanDoubleSpinBox2.setValue(35.)
        grid.addWidget(meanLabel2, 0, 0, 1, 1)
        grid.addWidget(self.meanDoubleSpinBox2, 0, 1, 1, 1)
        #
        stdLabel2 = QtWidgets.QLabel(self.dockWidget)
        stdLabel2.setText('Std Val')
        self.stdDoubleSpinBox2 = QtWidgets.QDoubleSpinBox(self.dockWidget)
        self.stdDoubleSpinBox2.setValue(35.)
        grid.addWidget(stdLabel2, 1, 0, 1, 1)
        grid.addWidget(self.stdDoubleSpinBox2, 1, 1, 1, 1)
        self.predictPathEdit = QtWidgets.QLineEdit(self.dockWidget)
        self.predictPathButton = QtWidgets.QPushButton(self.dockWidget)
        self.predictPathButton.clicked.connect(self.SetPredictFilePath)
        self.predictPathButton.setText('Open PTH')
        grid.addWidget(self.predictPathEdit, 2, 0, 1, 1)
        grid.addWidget(self.predictPathButton, 2, 1, 1, 1)
        self.rb21 = QRadioButton('CPU', self)
        self.rb22 = QRadioButton('GPU', self)
        self.bg2 = QButtonGroup(self)
        self.bg2.addButton(self.rb21, 1)
        self.bg2.addButton(self.rb22, 2)
        self.rb21.setChecked(True)
        grid.addWidget(self.rb21, 3, 0, 1, 1)
        grid.addWidget(self.rb22, 3, 1, 1, 1)
        self.predictButton = QPushButton(self.dockWidget)
        self.predictButton.setText('Predict')
        self.predictButton.clicked.connect(self.Predict)
        grid.addWidget(self.predictButton, 4, 1, 1, 1)

        self.retranslateUi()

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        #ImageAnnotation3D.setWindowTitle(_translate("ImageAnnotation3D", "3D图像标注"))
        self.imageGroupBox.setTitle(_translate("ImageAnnotationDock", "Image Option"))
        self.label_2.setText(_translate("ImageAnnotationDock", "Current Page"))
        self.mode_cbox.setItemText(0, _translate("ImageAnnotationDock", "Drag"))
        self.mode_cbox.setItemText(1, _translate("ImageAnnotationDock", "Draw"))
        self.mode_cbox.setItemText(2, _translate("ImageAnnotationDock", "Erase"))
        self.view_cbox.setItemText(0, _translate("ImageAnnotationDock", "Front"))
        self.view_cbox.setItemText(1, _translate("ImageAnnotationDock", "Side"))
        self.view_cbox.setItemText(2, _translate("ImageAnnotationDock", "Vertical"))
        self.label_3.setText(_translate("ImageAnnotationDock", "Operation"))
        self.label_1.setText(_translate("ImageAnnotationDock", "View Mode"))
        self.label_4.setText(_translate("ImageAnnotationDock", "Pen Width"))
        self.open_pb.setText(_translate("ImageAnnotationDock", "Open Image"))
        self.save_pb.setText(_translate("ImageAnnotationDock", "Save Label"))
        self.open_label.setText(_translate("ImageAnnotationDock", "Open Label"))
        self.split_orig.setText(_translate("ImageAnnotationDock", "Split Orig"))
        self.create_orig.setText(_translate("ImageAnnotationDock", "Create Orig"))
        self.create_bin.setText(_translate("ImageAnnotationDock", "Create Bin"))

    def SetTrainSavePath(self):
        self.trainSavePath = QFileDialog.getExistingDirectory(self, 'Save Directory', './')
        self.trainPathEdit.setText(self.trainSavePath+'/')

    def SetLRSamplePath(self):
        self.LRSamplePath = QFileDialog.getExistingDirectory(self, 'Orignal Directory', './')
        self.LRSamplePath = self.LRSamplePath
        self.LRSampleEdit.setText(self.LRSamplePath)

    def SetHRSamplePath(self):
        self.HRSamplePath = QFileDialog.getExistingDirectory(self, 'Label Directory', './')
        self.HRSamplePath = self.HRSamplePath
        self.HRSampleEdit.setText(self.HRSamplePath)

    def SetPredictFilePath(self):
        self.predictFilePath = QFileDialog.getOpenFileName(self, 'Open Train File', './', 'pth (*.pth)')
        self.predictFilePath = self.predictFilePath[0]
        self.predictPathEdit.setText(self.predictFilePath)

    def Train(self):
        if self.segmentor is None:
            QMessageBox.warning(self, 'code bug', 'please debug')

        if self.trainFlag:
            self.thread.terminate()
            self.trainButton.setText('train')
            self.trainFlag = False
            torch.cuda.empty_cache()
            self.segmentor.trainer = None
            return

        self.segmentor.SetTrainSavePath(self.trainPathEdit.text())
        self.segmentor.SetGPU(0)
        # python -m visdom.server
        # http://localhost:8097/
        self.trainFlag = True
        self.trainButton.setText('stop')
        self.thread = TrainThread()
        self.thread.SetEnv(self)
        self.thread.SetWorker(self.segmentor)
        self.thread.finished.connect(self.TrainFinish)
        self.thread.start()
        # if 0 == retVal:
        #     return
        # else:
        #     QMessageBox.warning(self, "warning", "An error occured: %d" % retVal)

    def TrainFinish(self):
        QMessageBox.information(self,"information", "training finished")

    def Predict(self):
        if self.segmentor is None:
            QMessageBox.warning(self, 'code bug', 'please debug')
        self.segmentor.SetPredictFile(self.predictPathEdit.text())
        if self.bg2.checkedId() == 1:
            self.segmentor.SetCPU()
        else:
            self.segmentor.SetGPU(0)
        self.segmentor.RestoreNetwork()
        self.segmentor.SetMeanVal(self.meanDoubleSpinBox2.value())
        self.segmentor.SetStdVal(self.stdDoubleSpinBox2.value())
        mask = self.segmentor.Predict(self.dataManager.images)
        if mask is not None:
            self.dataManager.mask = mask
            self.dataManager.cur_mask = mask[0,:]
            self.frame_sbox.setValue(0)
            view_idx = self.singleDock.view_cbox.currentIndex()
            self.showImageChange(view_idx, 0)
            return
        else:
            QMessageBox.warning(self, "warning", "An error occured in prediction" )
