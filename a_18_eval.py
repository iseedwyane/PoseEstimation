import torch
from AutoEncoder import AE
import torch.nn.functional as F
import torch.utils.data
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# 假设模型已经加载
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# 图像路径
image_paths = {
    'left': '/home/sen/Documents/InHand_pose/IMG_DATA_LS/IMG_DATA_MASTERBALL_0907/IMG_LEFT/00000000.jpg',
    'right': '/home/sen/Documents/InHand_pose/IMG_DATA_LS/IMG_DATA_MASTERBALL_0907/IMG_RIGHT/00000000.jpg',
    'behind': '/home/sen/Documents/InHand_pose/IMG_DATA_LS/IMG_DATA_MASTERBALL_0907/IMG_BEHIND/00000000.jpg'
}

# 图像预处理：调整大小，转换为张量，并标准化（如果需要）
transform = transforms.Compose([
    transforms.Resize((320, 320)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化（根据需要）
])

# 加载图像并应用预处理
img_left = transform(Image.open(image_paths['left'])).unsqueeze(0)  # Add batch dimension
img_right = transform(Image.open(image_paths['right'])).unsqueeze(0)
img_behind = transform(Image.open(image_paths['behind'])).unsqueeze(0)

# 合并图像：将三张图像合并成一个批次 (1, 3, H, W)
input_tensor = torch.cat((img_left, img_right, img_behind), dim=1)  # 拼接成 (1, 3C, H, W)

# 送到设备（GPU/CPU）
input_tensor = input_tensor.to(device)  # 设备可以是 "cuda" 或 "cpu"

# 加载预训练模型
reconstruction_model = AE(1, 1).to(device)  # 假设模型是 AE(1, 1)
model_path = './weights/reconstruction_weight_LS_64_3bjects_240912.pth'
reconstruction_model.load_state_dict(torch.load(model_path))  # 加载权重
reconstruction_model.eval()  # 设置模型为评估模式

# 执行推理
with torch.no_grad():  # 不计算梯度
    output = reconstruction_model(input_tensor)

# 假设我们使用 MSELoss 来计算输出与输入的差异
loss = torch.nn.MSELoss()(output, input_tensor)
print(f'Inference Loss: {loss.item()}')

# 输出张量需要转换回图像
output_image = output.squeeze(0).cpu().numpy()  # 移除批次维度并转回numpy格式
output_image = output_image.transpose(1, 2, 0)  # 调整维度顺序为 (H, W, C)

# 可视化
plt.imshow(output_image)
plt.show()