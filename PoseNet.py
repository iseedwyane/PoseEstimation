from collections import OrderedDict
from turtle import forward
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data
import resnet
from base import BaseModel
from AutoEncoder import AE
from LoadData import ReconstDataset
import copy


latent_num = 64
class EncordFun(AE):
    """A fully-convolutional network with an encoder-decoder architecture.
    """
    # hid_dims = [16, 32, 64, 128]
    # latent_num = 128

    def __init__(self):
        super().__init__(1,1)

    def forward(self, x):
        # print("raw img shape:",x.shape)
        out_enc = self._encoder(x)
        kn, h, w = out_enc.shape[1],out_enc.shape[2],out_enc.shape[3]
        # print("encoder feature map shape:",out_enc.shape)
        out_enc = torch.flatten(out_enc, start_dim=1)
        # print("encoder flat shape:",out_enc.shape )
        latent_vector = self._encoder_MPL(out_enc)
        return latent_vector


class ResMLP(nn.Module):
    def __init__(self, input_dimens) -> None:
        super().__init__()
        self.liner1 = nn.Linear(input_dimens, 100)
        self.norm = nn.BatchNorm1d(100)
        self.relu = nn.ReLU()
        self.liner2 = nn.Linear(100,input_dimens)

    def forward(self, x):
        residual = x

        out = self.liner1(x)
        # out = self.norm(out)
        out = self.relu(out)
        out = self.liner2(out)

        out += residual
        out = self.relu(out)
        return out

class PNet(BaseModel):

    def __init__(self, pre_path = "xx.pth"):
        super().__init__()
        self.weight_path = pre_path

        self.pose_MPL = nn.Sequential(
            OrderedDict([
                ("h1", torch.nn.Linear(latent_num*2+1, 200),),
                ("h1norm", torch.nn.BatchNorm1d(200, track_running_stats=True, affine=True)),
                ("active", torch.nn.ReLU(),),

                ("h2", torch.nn.Linear(200, 200),),
                ("h2norm", torch.nn.BatchNorm1d(200, track_running_stats=True, affine=True)),
                ("h2active", torch.nn.ReLU(),),

                ("h3", torch.nn.Linear(200, 100),),
                ("h3norm", torch.nn.BatchNorm1d(100, track_running_stats=True, affine=True)),
                ("h3active", torch.nn.ReLU(),),
                

                ("h4", torch.nn.Linear(100, 100),),
                ("h4norm", torch.nn.BatchNorm1d(100)),
                ("h4active", torch.nn.ReLU(),),
 
                ("h5", torch.nn.Linear(100, 100),),
                ("h5norm", torch.nn.BatchNorm1d(100)),
                ("h5active", torch.nn.ReLU(),),

                ("h6", torch.nn.Linear(100, 6),),

                # ("res1", ResMLP(latent_num*2+1),),
                # ("res2", ResMLP(latent_num*2+1),),
                # ("res3", ResMLP(latent_num*2+1),),

                # ("h6", torch.nn.Linear(latent_num*2+1, 6),),

                ])
        )

        # 
        self.encoder_model = EncordFun()
        
        self._init_weights()

        self.warmModel()

    def forward(self,img_left, img_right, joint_msg):
        # encoded image vector
        left = self.encoder_model(img_left)
        right = self.encoder_model(img_right)


        # print(left.shape, right.shape)
        # vector stack
        ks = torch.cat((left, right, joint_msg),axis = 1)#
        #print(ks.shape)#20240708lxb:torch.Size([1, 129])
        #print("====shape=====")#20240708lxb
        pose = self.pose_MPL(ks)
        #print("====shapeX=====")#20240708lxb
        if torch.any(torch.isnan(pose)):
            print("pose:", pose)
            print("input:", ks)
            raise Exception("Nan output")
        return pose
    
    def _init_weights(self):
        for m in self.pose_MPL:
            if isinstance(m, nn.Linear):
                # print('ss', m)
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            # elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)


    def warmModel(self):
        # Warmstarting Model Using Parameters from a Different Model and 
        # freeze the encoder, and only train the regression
        check_points = torch.load(self.weight_path)
        self.encoder_model.load_state_dict(check_points, strict=False)
        self.encoder_model._encoder.requires_grad_(False)
        self.encoder_model._encoder_MPL.requires_grad_(False)
        self.pose_MPL.requires_grad_(True)
        self.encoder_model.eval()

               
        # for child in self.encoder_model.children():
        #     for ii in range(len(child)):
        #         if type(child[ii])==nn.BatchNorm2d:
        #             child[ii].track_running_stats = False
        #             child[ii].momentum = 0.1
        

        # or assign each layers from loading model
        # check_points = torch.load(self.weight_path)
        # print(check_points.keys())
        # self.model._encoder.enc_conv0.weight = nn.Parameter(check_points["_encoder.enc_conv0.weight"])
        # self.model._encoder.enc_conv0.bias = nn.Parameter(check_points.state_dict()["_encoder.enc_conv0.bias"])



if __name__ == "__main__":
    xx = PNet("./weights/reconstruction_weight_LS_64_BSBPC.pth")
    print(xx)
    test_input = torch.ones(1, 1, 320, 320)

    ppx = xx.state_dict()
    # print("hh", ppx["pose_MPL.h1.weight"])

    print("number of parameters",xx.num_params)
    print("ss:", xx.parameters())
    print(xx.pose_MPL.named_parameters)
    print(xx.pose_MPL.h1.weight)
    # print(xx._encoder.enc_conv0.bias)
    # print(xx._encoder.parameters)
    # for n,x in xx._encoder.named_parameters():
    #     print(n,x.shape, x.requires_grad)