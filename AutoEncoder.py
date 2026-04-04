# MIT License.
# Copyright (c) 2022 by BioicDL. All rights reserved.
# Created by LiuXb on 2022/09/01
# -*- coding:utf-8 -*-

"""
@Modified: 
@Description:
"""
from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data
import resnet
from base import BaseModel
from LoadData import ReconstDataset


# input image resize to 320*320
class AE(BaseModel):
    """A fully-convolutional network with an encoder-decoder architecture.
    """
    # hid_dims = [64, 128,256,512]
    hid_dims = [16, 32, 64, 128]
    latent_num = 64
    encoder_flat_size = int(320*320/4/4)

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels

        self._encoder = nn.Sequential(
            OrderedDict(
                [
                    (
                        "enc_conv0",
                        nn.Conv2d(
                            in_channels,
                            self.hid_dims[0],
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    ("enc_norm0", nn.BatchNorm2d(self.hid_dims[0])),
                    ("enc_relu0", nn.ReLU(inplace=True)),
                    ("enc_pool1", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                    (
                        "enc_resn2",
                        resnet.BasicBlock(
                            self.hid_dims[0],
                            self.hid_dims[1],
                            downsample=nn.Sequential(
                                nn.Conv2d(self.hid_dims[0], self.hid_dims[1], kernel_size=1, bias=True),
                                nn.BatchNorm2d(self.hid_dims[1]),
                            ),
                            dilation=1,
                        ),
                    ),
                    ("enc_pool3", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                    (
                        "enc_resn4",
                        resnet.BasicBlock(
                            self.hid_dims[1],
                            self.hid_dims[2],
                            downsample=nn.Sequential(
                                nn.Conv2d(self.hid_dims[1], self.hid_dims[2], kernel_size=1, bias=True),
                                nn.BatchNorm2d(self.hid_dims[2]),
                            ),
                            dilation=1,
                        ),
                    ),
                    (
                        "enc_resn5",
                        resnet.BasicBlock(
                            self.hid_dims[2],
                            self.hid_dims[3],
                            downsample=nn.Sequential(
                                nn.Conv2d(self.hid_dims[2], self.hid_dims[3], kernel_size=1, bias=True),
                                nn.BatchNorm2d(self.hid_dims[3]),
                            ),
                            dilation=1,
                        ),
                    ),
                    (
                        "enc_conv1",
                        nn.Conv2d(
                            self.hid_dims[3],
                            1,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=True,
                        ),
                    ),
                    ("enc_norm1", nn.BatchNorm2d(1),),
                    ("enc_relu1", nn.ReLU(inplace=True),),
                ]
            )
        )


        self._decoder = nn.Sequential(
            OrderedDict(
                [
                    (
                        "dec_conv0",
                        nn.Conv2d(
                            1,
                            self.hid_dims[3],
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    ("dec_norm0", nn.BatchNorm2d(self.hid_dims[3]),),
                    ("dec_relu0", nn.ReLU(inplace=True),),
                    (
                        "dec_resn0",
                        resnet.BasicBlock(
                            self.hid_dims[3],
                            self.hid_dims[2],
                            downsample=nn.Sequential(
                                nn.Conv2d(self.hid_dims[3], self.hid_dims[2], kernel_size=1, bias=True),
                                nn.BatchNorm2d(self.hid_dims[2]),
                            ),
                            dilation=1,
                        ),
                    ),
                    (
                        "dec_resn1",
                        resnet.BasicBlock(
                            self.hid_dims[2],
                            self.hid_dims[1],
                            downsample=nn.Sequential(
                                nn.Conv2d(self.hid_dims[2], self.hid_dims[1], kernel_size=1, bias=True),
                                nn.BatchNorm2d(self.hid_dims[1]),
                            ),
                            dilation=1,
                        ),
                    ),
                    (
                        "dec_upsm2",
                        Interpolate(
                            scale_factor=2, mode="bilinear", align_corners=True
                        ),
                    ),
                    (
                        "dec_resn3",
                        resnet.BasicBlock(
                            self.hid_dims[1],
                            self.hid_dims[0],
                            downsample=nn.Sequential(
                                nn.Conv2d(self.hid_dims[1], self.hid_dims[0], kernel_size=1, bias=True),
                                nn.BatchNorm2d(self.hid_dims[0]),
                            ),
                            dilation=1,
                        ),
                    ),
                    (
                        "dec_upsm4",
                        Interpolate(
                            scale_factor=2, mode="bilinear", align_corners=True
                        ),
                    ),
                    (
                        "dec_conv5",
                        nn.Conv2d(self.hid_dims[0], out_channels, kernel_size=1, stride=1, padding=0,bias=True),
                    ),
                    ("dec_active", nn.Sigmoid(),),
                ]
            )
        )

        # self._encoder_MPL = nn.Sequential(
        #     OrderedDict([
        #         ("enc_h1", nn.Linear(self.hid_dims[-1]*self.encoder_flat_size, 200),),
        #         ("enc_h1active", nn.LeakyReLU(),),
        #         ("enc_h2", nn.Linear(200, 200),),
        #         ("enc_h2active", torch.nn.LeakyReLU(),),
        #         ("enc_h3", nn.Linear(200, self.latent_num),),
        #         ])
        # )

        self._encoder_MPL = nn.Sequential(
            OrderedDict([
                ("enc_h1", nn.Linear(1*self.encoder_flat_size, self.latent_num),),
                ])
        )

        self._decoder_MPL = nn.Sequential(
            OrderedDict([
                ("denc_h1", torch.nn.Linear(self.latent_num, 1*self.encoder_flat_size),),
                ])
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # if m.bias is not None:
                    # nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print("raw img shape:",x.shape)
        out_enc = self._encoder(x)
        kn, h, w = out_enc.shape[1],out_enc.shape[2],out_enc.shape[3]
        # print("encoder feature map shape:",out_enc.shape)
        out_enc = torch.flatten(out_enc, start_dim=1)

        latent_vector = self._encoder_MPL(out_enc)
        in_dec = self._decoder_MPL(latent_vector)

        in_dec = in_dec
        in_dec = in_dec.view(-1, kn, h, w)
        out_dec = self._decoder(in_dec)
        # print("reconstructured shape:", out_dec.shape)
        
        return out_dec


class Interpolate(nn.Module):
    def __init__(
            self,
            size=None,
            scale_factor=None,
            mode="nearest",
            align_corners=None,
    ):
        super().__init__()

        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, input):
        return F.interpolate(
            input, self.size, self.scale_factor, self.mode, self.align_corners
        )


if __name__ == "__main__":
    xx = AE(1, 1)
    test_input = torch.ones(1, 1, 320, 320)

    print(test_input)
    xx.forward(test_input)

    print(xx.num_params)
    print(xx.parameters())
    

    # ww = ReconstDataset(data_path=["/home/bionicdl-saber/Documents/GitHub/finger_6d_pose/data"], scaled_width = 320, scale_height= 180)
    # m = torch.utils.data.DataLoader(dataset=ww, batch_size=4, shuffle=True, num_workers=2, drop_last=True, pin_memory=False, collate_fn=None)
    # for i, datax in enumerate(m):
    #     outputs = xx(datax)
    #     print("recons shape",outputs.shape)


