import torch
import numpy as np
from models import CommonUNet
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
        self.pretrained_net = CommonUNet()
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
        zLen = origImg.shape[0]
        for ind in range(zLen):
            print('process %d'%ind)
            img = origImg[ind, :]
            resImg = np.zeros((origImg.shape[1], origImg.shape[2]), dtype=np.float)
            lowImg = np.array(img, dtype=np.float32)
            lowImg = (lowImg - self.meanVal) / (self.stdVal)
            lowImg = np.expand_dims(lowImg, axis=0)
            lowImg = np.expand_dims(lowImg, axis=0)
            lowImg = torch.from_numpy(lowImg).float()
            lowImg = lowImg.cuda(0)
            pre2 = self.pretrained_net(lowImg)
            saveImg = pre2.cpu().data.numpy()[0, :, :]
            resImg = saveImg
            resImg = (resImg - np.min(resImg)) / (np.max(resImg) - np.min(resImg)) * 254
            mask[ind, :] = resImg.astype(np.uint8)

        mask[mask > 125] = 255
        mask[mask <= 125] = 0
        return mask

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

