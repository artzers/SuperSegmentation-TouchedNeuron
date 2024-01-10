# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
from CLSTM import BDCLSTM, MyCLSTM


class UNetSmall(nn.Module):
    def __init__(self, num_channels=1, num_classes=2):
        super(UNetSmall, self).__init__()
        num_feat = [32,64,64]#[32, 64, 64, 64] #[32, 64, 128, 256]

        #wn = lambda x: torch.nn.utils.weight_norm(x)

        self.down1 = nn.Sequential((Conv3x3Small(num_channels, num_feat[0])))

        self.down2 = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                   #nn.BatchNorm2d(num_feat[0]),
                                   Conv3x3Small(num_feat[0], num_feat[1]))

        self.down3 = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                   #nn.BatchNorm2d(num_feat[1]),
                                   Conv3x3Small(num_feat[1], num_feat[2]))
        #
        # self.bottom = nn.Sequential(nn.MaxPool2d(kernel_size=2),
        #                             nn.BatchNorm2d(num_feat[2]),
        #                             Conv3x3Small(num_feat[2], num_feat[3]),
        #                             nn.BatchNorm2d(num_feat[3]))
        #
        self.up1 = UpSampleCat(num_feat[2], num_feat[1])
        self.upconv1 = nn.Sequential(Conv3x3Small(num_feat[2] + num_feat[1], num_feat[1]),
                                     #nn.BatchNorm2d(num_feat[1])
                                     )

        self.up2 = UpSampleCat(num_feat[1], num_feat[0])
        self.upconv2 = nn.Sequential(Conv3x3Small(num_feat[1] + num_feat[0], num_feat[0]),
                                     #nn.BatchNorm2d(num_feat[0])

                                     )
        #
        # self.up3 = UpSampleCat(num_feat[1], num_feat[0])
        # self.upconv3 = nn.Sequential(Conv3x3Small(num_feat[1] + num_feat[0], num_feat[0]),
        #                              nn.BatchNorm2d(num_feat[0]))


        # self.down3 = nn.Sequential(nn.MaxPool2d(kernel_size=2),
        #
        #                            (Conv3x3Small(num_feat[1], num_feat[2])))
        #
        # self.bottom = nn.Sequential(nn.MaxPool2d(kernel_size=2),
        #                             (Conv3x3Small(num_feat[2], num_feat[3])))
        #
        # self.up1 = UpSampleCat(num_feat[3], num_feat[2])
        # self.upconv1 = nn.Sequential((Conv3x3Small(num_feat[3] + num_feat[2], num_feat[2])),
        #                              )
        #
        # self.up2 = UpSampleCat(num_feat[2], num_feat[1])
        # self.upconv2 = nn.Sequential((Conv3x3Small(num_feat[2] + num_feat[1], num_feat[1])),
        #                              )
        #
        # self.up3 = UpSampleCat(num_feat[1], num_feat[0])
        # self.upconv3 = nn.Sequential((Conv3x3Small(num_feat[1] + num_feat[0], num_feat[0])),
        #                              )

        self.final = nn.Sequential(nn.Conv2d(num_feat[0],
                                             1,
                                             kernel_size=1),
                                   nn.Sigmoid())

    def forward(self, inputs, return_features=False):
        # print(inputs.data.size())
        down1_feat = self.down1(inputs)
        # print(down1_feat.size())
        down2_feat = self.down2(down1_feat)
        # print(down2_feat.size())
        down3_feat = self.down3(down2_feat)
        # print(down3_feat.size())
        # bottom_feat = self.bottom(down3_feat)

        # print(bottom_feat.size())
        up1_feat = self.up1(down3_feat, down2_feat)
        # print(up1_feat.size())
        up1_feat = self.upconv1(up1_feat)
        # print(up1_feat.size())
        up2_feat = self.up2(up1_feat, down1_feat)
        # print(up2_feat.size())
        up2_feat = self.upconv2(up2_feat)
        # print(up2_feat.size())
        # up3_feat = self.up3(up2_feat, down1_feat)
        # print(up3_feat.size())
        # up3_feat = self.upconv3(up3_feat)
        # print(up3_feat.size())

        if return_features:
            outputs = up2_feat
        else:
            outputs = self.final(up2_feat)

        return outputs


class Conv3x3Small(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(Conv3x3Small, self).__init__()
        # wn = lambda x: torch.nn.utils.weight_norm(x)
        self.conv1 = nn.Sequential((nn.Conv2d(in_feat, out_feat,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1)),
                                   # nn.BatchNorm2d(out_feat),
                                   nn.InstanceNorm2d(out_feat, affine= True),
                                   nn.ReLU()
                                   )

        # self.conv2 = nn.Sequential(nn.Conv2d(out_feat, out_feat,
        #                                      kernel_size=3,
        #                                      stride=1,
        #                                      padding=1),
        #                            nn.ELU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        # outputs = self.conv2(outputs)
        return outputs


class UpConcat(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(UpConcat, self).__init__()

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # self.deconv = nn.ConvTranspose2d(in_feat, out_feat,
        #                                  kernel_size=3,
        #                                  stride=1,
        #                                  dilation=1)

        self.deconv = nn.ConvTranspose2d(in_feat,
                                         out_feat,
                                         kernel_size=2,
                                         stride=2)

    def forward(self, inputs, down_outputs):
        # TODO: Upsampling required after deconv?
        # outputs = self.up(inputs)
        outputs = self.deconv(inputs)
        out = torch.cat([down_outputs, outputs], 1)
        return out


class UpSampleCat(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(UpSampleCat, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv = nn.Conv2d(in_feat,
                                         in_feat, # here !!
                                         kernel_size=3,
                                        padding=1)

    def forward(self, inputs,down_outputs):
        # TODO: Upsampling required after deconv?
        outputs = self.up(inputs)
        out = self.conv(outputs)
        out = torch.cat([out, down_outputs], 1)
        return out


class UpSample(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(UpSample, self).__init__()

        self.up = nn.Upsample(scale_factor=3, mode='bilinear')

        self.conv = nn.Conv2d(in_feat,
                                         out_feat,
                                         kernel_size=3,
                                        padding=1)

    def forward(self, inputs):
        # TODO: Upsampling required after deconv?
        outputs = self.up(inputs)
        out = self.conv(outputs)
        return out

# class CommonSRCLSN(nn.Module):
#     def __init__(self):
#         super(CommonSRCLSN, self).__init__()
#         # wn = lambda x: torch.nn.utils.weight_norm(x)
#         self.unet = UNetSmall()
#         # self.conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
#         self.srclsn = SRCLSN(input_channels=32, hidden_channels=[64,64])#[32,32]
#         self.conv = nn.Sequential(
#             (nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)),
#             # nn.BatchNorm2d(32),
#             nn.InstanceNorm2d(32, affine= True),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, padding=1))
#         #self.outputSize = 30
#         #self.up = UpSample(32, self.outputSize)
#         #self.tail = nn.Conv2d(in_channels=16*32, out_channels=48, kernel_size=3, padding=1)
#         self._init_weight()
#
#     def _init_weight(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv3d):
#                 nn.init.kaiming_normal_(m.weight)
#             elif isinstance(m, nn.Linear):
#                 m.weight.data.normal(0, 0.01)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm3d):
#                 torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
#                 torch.nn.init.constant_(m.bias.data, 0.0)
#             elif isinstance(m, nn.InstanceNorm2d):
#                 nn.init.constant_(m.bias, 0.0)
#                 nn.init.normal_(m.weight, 1.0, 0.02)
#             elif isinstance(m, nn.GroupNorm):
#                 torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
#                 torch.nn.init.constant_(m.bias.data, 0.0)
#
#     def forward(self, input):
#         #input batch x
#         #input = input.unsqueeze(0)
#         res = []
#         for k in range(input.shape[1]):
#             x = self.unet(input[:, k, :], return_features=True)
#             if k == 0:
#                 self.srclsn.init_hidden(x)
#             _ = self.srclsn(x)
#
#         for k in range(input.shape[1]):
#             x = self.unet(input[:,k,:], return_features=True) #:,15,1,:,: -> 1,1,64,:,:
#             output = self.srclsn(x)
#             res.append(self.conv(output))
#         # output = self.lstm(input)
#         output = torch.cat(res, dim=1)
#         # output = self.tail(res)
#         return torch.sigmoid(output)

class BDCLSTMSegNet(nn.Module):
    def __init__(self):
        super(BDCLSTMSegNet, self).__init__()
        # wn = lambda x: torch.nn.utils.weight_norm(x)
        self.unet = UNetSmall()
        # self.conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.lstm = BDCLSTM(input_channels=32, hidden_channels=[64,64])#[32,32]
        self.conv = nn.Sequential(
            (nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)),
            # nn.BatchNorm2d(32),
            nn.InstanceNorm2d(32, affine= True),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, padding=1))
        #self.outputSize = 30
        #self.up = UpSample(32, self.outputSize)
        #self.tail = nn.Conv2d(in_channels=16*32, out_channels=48, kernel_size=3, padding=1)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.bias, 0.0)
                nn.init.normal_(m.weight, 1.0, 0.02)
            elif isinstance(m, nn.GroupNorm):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)

    def forward(self, input):
        #input batch x
        res = []
        for k in range(input.shape[1]):
            x = self.unet(input[:, k, :], return_features=True)
            if k == 0:
                self.lstm.init_hidden(x)
            _ = self.lstm(x)

        for k in range(input.shape[1]):
            x = self.unet(input[:,k,:], return_features=True) #:,15,1,:,: -> 1,1,64,:,:
            output = self.lstm(x)
            res.append(self.conv(output))
        # output = self.lstm(input)
        output = torch.cat(res, dim=1)
        # output = self.tail(res)
        return torch.sigmoid(output)
