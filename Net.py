import os, torch, tifffile
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler as lrs
import torch.nn.utils as utils
from torch.utils.checkpoint import checkpoint

from scipy.ndimage.interpolation import zoom
import itertools
from torch.autograd import Variable
import torch.autograd as autograd
from models import CommonUNet
from losses import DICELoss
from Util import RestoreNetImg, SoftDiceLoss

logger = None
globalMean = 0
globalStd = 0

class Trainer:
    def __init__(self,
                 data_loader,
                 test_loader,
                 scheduler=lrs.StepLR,
                 dev='cuda:0', devid=0):
        self.dataLoader = data_loader
        self.testLoader = test_loader
        # self.scheduler = scheduler(
        #     self.optimizer, step_size=2000, gamma=0.8, last_epoch=-1)
        self.dev = dev
        self.cudaid = devid

        self.precise_dice_loss = SoftDiceLoss()
        self.precise_dice_loss = self.precise_dice_loss.cuda(self.cudaid)


        # Loss weights
        self.lambda_adv = 1
        self.lambda_cycle = 10
        self.lambda_gp = 10

        # Initialize generator and discriminator
        self.G_AB = CommonUNet()

        self.G_AB.cuda(self.cudaid)

        # Optimizers
        self.optimizer_G = torch.optim.Adam([{'params': self.G_AB.parameters(),  \
                                              'initial_lr': 0.002}], lr=0.002)

        # self.optimizer_G = torch.optim.SGD([{'params': self.G_AB.parameters(), \
        #                                       'initial_lr': 0.00001}], lr=0.00001, momentum=0.9)


        self.scheduler_G = scheduler(self.optimizer_G, step_size=2000, gamma=0.9, last_epoch=-1)
        #self.scheduler_G_AB = scheduler(self.optimizer_G_AB, step_size=10000, gamma=0.9, last_epoch=-1)#36000


    def Train(self, turn=2):
        self.shot = -1
        torch.set_grad_enabled(True)

        for t in range(turn):

            # if self.gclip > 0:
            #     utils.clip_grad_value_(self.net.parameters(), self.gclip)

            for kk, (img, mask) in enumerate(self.dataLoader):#have been normalized
                # torch.cuda.empty_cache()
                self.shot = self.shot + 1
                self.scheduler_G.step()

                img = img.cuda(self.cudaid)
                #img = img.unsqueeze(0)
                mask = mask.cuda(self.cudaid)
                #mask = mask.unsqueeze(0)

                if True:
                    self.optimizer_G.zero_grad()

                    # Generate a batch of images
                    seg_A = self.G_AB(img)

                    seg_loss = self.precise_dice_loss(seg_A, mask)
                    seg_loss.backward()
                    self.optimizer_G.step()

                if self.shot % 50 == 0:

                    lr = self.scheduler_G.get_lr()[0]
                    print("\r[Epoch %d] [Batch %d] [LR:%f][Dice loss: %f]"
                                            % (
                                                t,
                                                self.shot,
                                                lr,
                                                seg_loss.item(),
                                            )
                                        )

                    # reImg = (seg_A).cpu().data.numpy()[0, :, :, :]
                    # reImg2XY = np.max(reImg, axis=0)
                    # reImg2XY = (reImg2XY * 254).astype(np.uint8)
                    # logger.img('segXY', reImg2XY)
                    # # interpolate
                    # origDisplay = img.cpu().data.numpy()[0, :, :, :]
                    # origDisplay = RestoreNetImg(origDisplay, globalMean, globalStd)
                    # origDisplay = (np.max(origDisplay, axis=0)).astype(np.uint8)
                    # logger.img('origXY', origDisplay)
                    # labelXY = np.max(mask.cpu().data.numpy()[0, :, :, :], axis=0)
                    # labelXY = RestoreNetImg(labelXY, 0, 1)
                    # logger.img('labelXY', labelXY)
                    # lossVal = np.float(seg_loss.cpu().data.numpy())
                    reImg = (seg_A).cpu().data.numpy()[0, :, :, :]
                    reImg2XY = np.max(reImg, axis=0)
                    reImg2XY = (reImg2XY * 254).astype(np.uint8)
                    logger.img('segImg2XY', reImg2XY)
                    reImg2XZ = np.max(reImg, axis=1)
                    reImg2XZ = (reImg2XZ * 254).astype(np.uint8)
                    logger.img('segImg2XZ', reImg2XZ)

                    lrImg = img.cpu().data.numpy()[0, :, :, :]
                    zoom2 = RestoreNetImg(lrImg, globalMean, globalStd)
                    zoom2XY = np.max(zoom2, axis=0)
                    logger.img('lrImgXY', zoom2XY)
                    zoom2XZ = np.transpose(np.max(zoom2, axis=2), (1, 0, 2))
                    logger.img('lrImgXZ', zoom2XZ)

                    highImg = mask.cpu().data.numpy()[0, :, :, :]
                    highImgXY = np.max(highImg, axis=0)
                    highImgXY = (highImgXY * 255.)  # RestoreNetImg(highImgXY, 0, 1)
                    logger.img('binImgXY', highImgXY)
                    highImgXZ = np.max(highImg, axis=1)
                    highImgXZ = (highImgXZ * 255.)  # RestoreNetImg(highImgXY, 0, 1)
                    logger.img('binImgXZ', highImgXZ)
                    lossVal = np.float(seg_loss.cpu().data.numpy())
                    if np.abs(lossVal) > 1:
                        print('G loss > 1')
                    elif lossVal < 0:
                        print('G loss < 0')
                    else:
                        logger.plot('G_loss', lossVal)

                if self.shot != 0 and self.shot % 800 == 0:
                    if not os.path.exists('saved_models/'):
                        os.mkdir('saved_models/')
                    torch.save(self.G_AB.state_dict(), "saved_models/common_%d.pth" % ( self.shot))





