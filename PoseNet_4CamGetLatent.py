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
    """A fully-convolutional network with an encoder-decoder architecture."""
    def __init__(self):
        super().__init__(1, 1)

    def forward(self, x):
        out_enc = self._encoder(x)
        kn, h, w = out_enc.shape[1], out_enc.shape[2], out_enc.shape[3]
        out_enc = torch.flatten(out_enc, start_dim=1)
        latent_vector = self._encoder_MPL(out_enc)
        return latent_vector

class PNet(BaseModel):
    def __init__(self, pre_path="xx.pth"):
        super().__init__()
        self.weight_path = pre_path

        self.pose_MPL = nn.Sequential(
            OrderedDict([
                ("h1", torch.nn.Linear(latent_num*3+1, 200)),
                ("h1norm", torch.nn.BatchNorm1d(200, track_running_stats=True, affine=True)),
                ("active", torch.nn.ReLU()),

                ("h2", torch.nn.Linear(200, 200)),
                ("h2norm", torch.nn.BatchNorm1d(200, track_running_stats=True, affine=True)),
                ("h2active", torch.nn.ReLU()),

                ("h3", torch.nn.Linear(200, 100)),
                ("h3norm", torch.nn.BatchNorm1d(100, track_running_stats=True, affine=True)),
                ("h3active", torch.nn.ReLU()),

                ("h4", torch.nn.Linear(100, 100)),
                ("h4norm", torch.nn.BatchNorm1d(100)),
                ("h4active", torch.nn.ReLU()),

                ("h5", torch.nn.Linear(100, 100)),
                ("h5norm", torch.nn.BatchNorm1d(100)),
                ("h5active", torch.nn.ReLU()),

                ("h6", torch.nn.Linear(100, 6)),
            ])
        )

        self.encoder_model = EncordFun()
        self._init_weights()
        self.warmModel()

        self.ks_list = []  # 用于保存 ks

    def forward(self, img_left, img_right, img_behind, joint_msg):
        # encoded image vector
        left = self.encoder_model(img_left)
        right = self.encoder_model(img_right)
        behind = self.encoder_model(img_behind)
        print("left:", left)
        print("left shape:", left.shape)#
        print("joint_msg:", joint_msg)
        print("joint_msg shape:", joint_msg.shape)#  
        # vector stack
        ks = torch.cat((left, right, behind, joint_msg), axis=1)
        
        # 将 ks 保存到列表中
        self.ks_list.append(ks.detach().cpu().numpy())  # detach() 防止梯度传播

        pose = self.pose_MPL(ks)
        if torch.any(torch.isnan(pose)):
            print("pose:", pose)
            print("input:", ks)
            raise Exception("Nan output")
        return pose

    def _init_weights(self):
        for m in self.pose_MPL:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def warmModel(self):
        check_points = torch.load(self.weight_path)
        self.encoder_model.load_state_dict(check_points, strict=False)
        self.encoder_model._encoder.requires_grad_(False)
        self.encoder_model._encoder_MPL.requires_grad_(False)
        self.pose_MPL.requires_grad_(True)
        self.encoder_model.eval()

    def save_ks(self, filename="./weights/ks.npy"):
        # 将 ks_list 转为 np.array 并保存为 .npy 文件
        ks_array = np.vstack(self.ks_list)  # 将 ks 拼接成一个 nx193 的数组
        np.save(filename, ks_array)
        print(f"KS saved to {filename}")

if __name__ == "__main__":
    xx = PNet("./weights/reconstruction_weight_VASE_250110.pth")
    print(xx)
    test_input = torch.ones(1, 1, 320, 320)

    ppx = xx.state_dict()
    print("number of parameters", xx.num_params)
    print("ss:", xx.parameters())

    # 模拟一个前向传播过程
    output = xx(test_input, test_input, test_input, test_input)
    
    # 保存 ks 到文件
    xx.save_ks("./weights/ks.npy")
