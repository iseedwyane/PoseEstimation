import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
from collections import OrderedDict

class VAE(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        # Encoder and Decoder definitions here...
        self.hid_dims = [16, 32, 64, 128]
        self.latent_num = 64
        self.encoder_flat_size = int(320*320/4/4)

        # Encoder network
        self._encoder = nn.Sequential(
            OrderedDict([
                ("enc_conv0", nn.Conv2d(in_channels, self.hid_dims[0], kernel_size=3, stride=1, padding=1, bias=True)),
                ("enc_norm0", nn.BatchNorm2d(self.hid_dims[0])),
                ("enc_relu0", nn.ReLU(inplace=True)),
                ("enc_pool1", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                # Add resnet layers
            ])
        )

        # Decoder network
        self._decoder = nn.Sequential(
            OrderedDict([
                ("dec_conv0", nn.Conv2d(1, self.hid_dims[3], kernel_size=3, stride=1, padding=1, bias=True)),
                ("dec_norm0", nn.BatchNorm2d(self.hid_dims[3])),
                ("dec_relu0", nn.ReLU(inplace=True)),
                # Add resnet layers
            ])
        )

        self._encoder_MPL = nn.Sequential(
            OrderedDict([("enc_h1", nn.Linear(1*self.encoder_flat_size, self.latent_num))])
        )

        self._decoder_MPL = nn.Sequential(
            OrderedDict([("denc_h1", torch.nn.Linear(self.latent_num, 1*self.encoder_flat_size))])
        )

    def encode(self, x):
        out_enc = self._encoder(x)
        out_enc = torch.flatten(out_enc, start_dim=1)
        latent_vector = self._encoder_MPL(out_enc)
        return latent_vector

    def forward(self, x):
        latent_vector = self.encode(x)
        in_dec = self._decoder_MPL(latent_vector)
        in_dec = in_dec.view(-1, 1, 320, 320)
        out_dec = self._decoder(in_dec)
        return out_dec

# Image preprocessing and loading
def load_and_preprocess_image(image_path, resize=(320, 320)):
    img = Image.open(image_path).convert("L")
    preprocess = transforms.Compose([transforms.Resize(resize), transforms.ToTensor()])
    return preprocess(img).unsqueeze(0)

# 读取图片并进行预处理
imgLeft_path = '/home/sen/Documents/InHand_pose/IMG_DATA_LS/IMG_DATA_PEACH_241226/IMG_LEFT/00000004.jpg'
imgRight_path = '/home/sen/Documents/InHand_pose/IMG_DATA_LS/IMG_DATA_PEACH_241226/IMG_RIGHT/00000004.jpg'
imgBehind_path = '/home/sen/Documents/InHand_pose/IMG_DATA_LS/IMG_DATA_PEACH_241226/IMG_BEHIND/00000004.jpg'

imgLeft = load_and_preprocess_image(imgLeft_path)
imgRight = load_and_preprocess_image(imgRight_path)
imgBehind = load_and_preprocess_image(imgBehind_path)

# Stack images into a batch
image_tensor = torch.cat((imgLeft, imgRight, imgBehind), dim=0)

pre_model_pth = "./weights/reconstruction_weight_PEACH_241226.pth"#1/2
# Load pre-trained model
vae = VAE()
vae.load_state_dict(torch.load(pre_model_pth))
vae.eval()

# Get latent vectors for the images
with torch.no_grad():
    latent_vector = vae.encode(image_tensor)
    print(latent_vector.shape)  # Output shape: (3, 64)
