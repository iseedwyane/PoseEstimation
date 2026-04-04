import torch
import cv2
import numpy as np
from PoseNet import PNet
from LoadData import ReconstDataset
from scipy.spatial.transform import Rotation as spR

class RealTimePrediction:
    def __init__(self):     
        self.pre_model_pth = "./weights/reconstruction_weight_LS_64_BSBPC.pth"
        self.width = 320
        self.height = 320
        self.batch_size = 32

        self.saved_pth = "./weights/pose_weight.pth"
        self.data_folder = "./IMG_DATA"#any path is ok, because training don't need this input.
        self.data_folder_var = "./IMG_DATA"

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        print("============__init__============")


    def var(self, varData , model = None, trained_weights = "./weights/pose_weight.pth"):
        # 载入测试集
        dataset_image = ReconstDataset(data_path=varData, scaled_width = self.width, scale_height= self.height)
        # print("sample numbers: ", len(dataset_image))
        var_data = torch.utils.data.DataLoader(dataset=dataset_image, batch_size=self.batch_size, shuffle=False, num_workers=8, drop_last=False, pin_memory=False, collate_fn=None)
        # 载入模型
        if model==None:
            # reload_model = PNet(self.pre_model_pth).to(self.device)
            reload_model = PNet(self.saved_pth).to(self.device)
            # if self.device.type != 'cpu':
            #     reload_model = torch.nn.DataParallel(reload_model, device_ids=[0,1])
            reload_model.load_state_dict(torch.load(trained_weights))        
        else:
            # reload_model = copy.deepcopy(model)
            reload_model = model
        # reload_model.eval()
        reload_model.pose_MPL.eval()

        vloss = torch.nn.MSELoss()

        for i, datax in enumerate(var_data):
            imgLeft= datax[0]
            imgRight= datax[1]
            joint_config = datax[2]
            objectPose = datax[3]

            imgs1= imgLeft.to(self.device)
            imgs2= imgRight.to(self.device)
            jc = joint_config.to(self.device)
            labels = objectPose.to(self.device)
            labels = labels.squeeze(1)

            #print(cnt, imgs.shape)
            #print(imgs.is_cuda)
            outputs = reload_model(imgs1, imgs2, jc)
            print(outputs.shape, outputs[:,0:3].shape)
            l1 = vloss(outputs[:,0:3], labels[:,0:3])
            l2 = vloss(outputs[:,3:6], labels[:,3:6])
            loss = 0.1*l1 + 10*l2
            print(0.1*l1,l2*3)
            loss = vloss(outputs, labels)

if __name__ == "__main__":
    saved_pth = ["./weights/pose_weight_baseball.pth","./weights/pose_weight_strawberry.pth","./weights/pose_weight_bottlecover.pth","./weights/pose_weight_peach.pth","./weights/pose_weight_cube.pth"]
    data_path = ["IMG_DATA_BASEBALL_0628","IMG_DATA_STARW_0628","IMG_DATA_BOTTLECOVER_0628","IMG_DATA_PEACH_0628","IMG_DATA_CUBE_0628"]
    print("============main__init__============")
    Predict_test = RealTimePrediction()
    print("============for============")
    for i in range(len(saved_pth)):
        Predict_test.saved_pth = saved_pth[i]

        Predict_test.data_folder = "./IMG_DATA_LS/" + data_path[i] + "/train.txt"
        Predict_test.data_folder_var = "./IMG_DATA_LS/" + data_path[i] + "/var.txt"     

        print("=================",data_path[i],"===================")    
        #Predict_test.var()
          
        x1 = Predict_test.var(Predict_test.data_folder_var,trained_weights = Predict_test.saved_pth)
        print("=================var===================")    
        print("Predicted Pose:", x1)
