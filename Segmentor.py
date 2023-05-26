import torch
import os
import tifffile
import numpy as np
from models import CommonSRCLSN
from vis import vis_tool
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler as lrs
import torch.nn.utils as utils
from torch.utils.checkpoint import checkpoint
import math,time
from Util import DataPackage
from Util import GetMemoryDataSetAndCrop, Get2DMemoryDataSetAndCrop,GetRNNMemoryDataSetAndCrop
import Net
'''
python -m visdom.server
http://localhost:8097/
'''

saveRoot = './out/test'

class Segmentor:
    def __init__(self):
        self.pretrained_net = None
        self.deviceStr = 'cpu'
        #type : 0 - cpu, 1 - cuda:0
        self.deviceType = 0
        self.netPath = None
        self.initFlag = False
        self.meanVal = 0
        self.stdVal = 0

    def SetTrainSavePath(self, netPath):
        self.netPath = netPath

    def SetPredictFile(self, netPath):
        self.netPath = netPath

    def RestoreNetwork(self, netPath=None):
        if netPath is not None:
            self.netPath = netPath
        if self.pretrained_net is not None:
            self.pretrained_net = None
        self.pretrained_net = CommonSRCLSN()
        self.pretrained_net.load_state_dict(
            torch.load(self.netPath, map_location=self.deviceStr))
        if self.deviceType == 1:
            self.pretrained_net = self.pretrained_net.cuda(0)
            self.pretrained_net.eval()
            torch.set_grad_enabled(False)
            torch.cuda.empty_cache()
        elif self.deviceType == 0:
            self.pretrained_net = self.pretrained_net.cpu()
            self.pretrained_net.eval()
            torch.set_grad_enabled(False)
        self.initFlag = True

    def SetCPU(self):
        if self.deviceType == 0:
            return
        # type : 0 - cpu, 1 - cuda:0
        self.deviceType = 0
        self.deviceStr = 'cpu'

    def SetGPU(self, index):
        if self.deviceType == index + 1:
            return
        # type : 0 - cpu, 1 - cuda:0
        self.deviceType = index+1
        self.deviceStr = 'cuda:'+'%d' % index

    def Clear(self):
        self.pretrained_net = None
        self.deviceStr = 'cpu'
        # type : 0 - cpu, 1 - cuda:0
        self.deviceType = 0
        self.netPath = None
        self.initFlag = False

    def SetMeanVal(self, val):
        self.meanVal = val

    def SetStdVal(self, val):
        self.stdVal = val

    def Predict(self, origImg):
        if self.initFlag == False:
            return None
        if self.pretrained_net is None:
            return None

        torch.set_grad_enabled(False)
        if self.deviceType == 1:
            torch.cuda.empty_cache()

        mask = np.zeros(origImg.shape, dtype=np.uint8)
        minLowRange = [0, 0, 0]
        minLowRange = minLowRange[-1::-1]
        xPad = 10
        yPad = 10
        zPad = 4
        # xPad = 0
        # yPad = 0
        # zPad = 0
        maxLowRange = [origImg.shape[2], origImg.shape[1], origImg.shape[0]]
        # print(maxLowRange)
        # maxLowRange=[120,120,32]
        maxLowRange = maxLowRange[-1::-1]  # reverse
        # print(maxLowRange)
        yMinLowList = []
        yMaxLowList = []
        xMinLowList = []
        xMaxLowList = []
        zMinLowList = []
        zMaxLowList = []

        for k in range(minLowRange[1], maxLowRange[1] - 119, 120 - 6 * yPad):
            yMinLowList.append(k)
            yMaxLowList.append(k + 120)  # 59

        if yMaxLowList[-1] < maxLowRange[1]:
            yMaxLowList.append(maxLowRange[1])
            yMinLowList.append(maxLowRange[1] - 120)

        for k in range(minLowRange[2], maxLowRange[2] - 119, 120 - 6 * xPad):
            xMinLowList.append(k)
            xMaxLowList.append(k + 120)

        if xMaxLowList[-1] < maxLowRange[2]:
            xMaxLowList.append(maxLowRange[2])
            xMinLowList.append(maxLowRange[2] - 120)

        for k in range(minLowRange[0], maxLowRange[0] - 31, 32 - 4 * zPad):
            zMinLowList.append(k)
            zMaxLowList.append(k + 32)

        if zMaxLowList[-1] < maxLowRange[0]:
            zMaxLowList.append(maxLowRange[0])
            zMinLowList.append(maxLowRange[0] - 32)

        zBase = zMinLowList[0]
        resImg = np.zeros((origImg.shape[0] * 3, origImg.shape[1], origImg.shape[2]), dtype=np.float)
        # padImg = np.zeros((img.shape[0] + 2*zPad,img.shape[1] + 2*yPad,img.shape[2] + 2*xPad ), dtype = img.dtype)
        # padImg[zPad:zPad+img.shape[0], yPad:yPad+img.shape[1], xPad:xPad+img.shape[2]] = img
        # tifffile.imwrite('./test/hehe.tif',padImg)
        # resImg = np.zeros((padImg.shape[0]*3, padImg.shape[1],padImg.shape[2]), dtype=np.float)
        for i in range(len(zMinLowList)):
            for j in range(len(yMinLowList)):
                for k in range(len(xMinLowList)):
                    print('processing :%d-%d, %d-%d %d-%d' % (xMinLowList[k], xMaxLowList[k],
                                                              yMinLowList[j], yMaxLowList[j],
                                                              zMinLowList[i], zMaxLowList[i]))
                    lowImg = origImg[zMinLowList[i]:zMaxLowList[i],
                             yMinLowList[j]:yMaxLowList[j],
                             xMinLowList[k]:xMaxLowList[k]]
                    lowImg = np.array(lowImg, dtype=np.float32)
                    lowImg = (lowImg - 30.) / (30.)
                    lowImg = np.expand_dims(lowImg, axis=1)
                    lowImg = np.expand_dims(lowImg, axis=0)
                    lowImg = torch.from_numpy(lowImg).float()
                    lowImg = lowImg.cuda(0)
                    pre2 = self.pretrained_net(lowImg)
                    saveImg = pre2.cpu().data.numpy()[0, :, :, :]
                    if ((k == 0 and j == 1 and i == 0) or (k == 2 and j == 1 and i == 0)
                            or (k == 0 and j == 1 and i == 1) or (k == 2 and j == 1 and i == 1)
                            or (k == 0 and j == 1 and i == 2) or (k == 2 and j == 1 and i == 2)):
                        resImg[zMinLowList[i] * 3:zMaxLowList[i] * 3,
                        yMinLowList[j] + yPad:yMaxLowList[j] - yPad,
                        xMinLowList[k]:xMaxLowList[k]] \
                            = saveImg[0:96, yPad:120 - yPad, 0:120]
                    elif ((k == 1 and j == 0 and i == 0) or (k == 1 and j == 2 and i == 0)
                          or (k == 1 and j == 0 and i == 1) or (k == 1 and j == 2 and i == 1)
                          or (k == 1 and j == 0 and i == 2) or (k == 1 and j == 2 and i == 2)):
                        resImg[zMinLowList[i] * 3:zMaxLowList[i] * 3,
                        yMinLowList[j]:yMaxLowList[j],
                        xMinLowList[k] + xPad:xMaxLowList[k] - xPad] \
                            = saveImg[0:96, 0:120, xPad:120 - xPad]
                    elif ((k == 1 and j == 1 and i == 0) or (k == 1 and j == 1 and i == 2)):
                        resImg[zMinLowList[i] * 3:zMaxLowList[i] * 3,
                        yMinLowList[j] + yPad:yMaxLowList[j] - yPad,
                        xMinLowList[k] + xPad:xMaxLowList[k] - xPad] \
                            = saveImg[0:96, yPad:120 - yPad, xPad:120 - xPad]
                    elif (k == 1 and j == 1 and i == 1):
                        resImg[zMinLowList[i] * 3 + 2 * zPad:zMaxLowList[i] * 3 - 2 * zPad,
                        yMinLowList[j] + yPad:yMaxLowList[j] - yPad,
                        xMinLowList[k] + xPad:xMaxLowList[k] - xPad] \
                            = saveImg[2 * zPad:96 - 2 * zPad, yPad:120 - yPad, xPad:120 - xPad]
                    else:
                        resImg[zMinLowList[i] * 3:zMaxLowList[i] * 3,
                        yMinLowList[j]:yMaxLowList[j],
                        xMinLowList[k]:xMaxLowList[k]] \
                            = saveImg[0:96, 0:120, 0:120]
                    # resImg[zMinLowList[i]*3+3*zPad:zMaxLowList[i]*3-3*zPad,
                    # yMinLowList[j]+yPad:yMaxLowList[j]-yPad,
                    # xMinLowList[k]+xPad:xMaxLowList[k]-xPad] \
                    # = saveImg[3*zPad:96-3*zPad, yPad:120-yPad, xPad:120-xPad]
                    # resImg[zMinLowList[i]*3:zMaxLowList[i]*3,
                    #                           yMinLowList[j]:yMaxLowList[j],
                    #                           xMinLowList[k]:xMaxLowList[k]]\
                    #     = np.maximum(saveImg, resImg[zMinLowList[i]*3:zMaxLowList[i]*3,
                    #                           yMinLowList[j]:yMaxLowList[j],
                    #                           xMinLowList[k]:xMaxLowList[k]])

        # resImg = resImg[3*zPad:3*zPad+img.shape[0]*3, yPad:yPad+img.shape[1], xPad:xPad+img.shape[2]]
        resImg = (resImg - np.min(resImg)) / (np.max(resImg) - np.min(resImg)) * 254
        savePath = os.path.join(saveRoot, "orig" + '_clstm.tif')
        print('save as %s' % savePath)
        tifffile.imwrite(savePath, np.uint8(resImg))  # , compress = 6
        return resImg

    def Train(self, lrDir, hrDir, mean=35., std=35., epoch=500, visdomable=False,
              initLR = 0.002, decayTurn = 2000, savePath = None):
        env = 'Common2DSeg'
        globalDev = self.deviceStr
        globalDeviceID = 0

        train_dataset = GetRNNMemoryDataSetAndCrop(lrDir, hrDir, [120, 120], 100,
                                                  mean, std)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
        if visdomable == True:
            logger = vis_tool.Visualizer(env=env)
            logger.reinit(env=env)
            Net.logger = logger
        if visdomable == False:
            logger = None
        self.trainer = Net.Trainer(data_loader=train_loader,
                              test_loader=None )
        Net.globalMean = mean
        Net.globalStd = std
        if savePath is not None:
            self.trainer.saveRoot = savePath
        self.trainer.initLR = initLR
        self.trainer.decayTurn = decayTurn
        time_start = time.time()
        self.trainer.Train(turn=epoch)
        time_end = time.time()
        print('totally time cost', time_end - time_start)

