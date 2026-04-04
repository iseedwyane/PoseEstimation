import numpy as np
import matplotlib.pyplot as plt

# 定义数据路径
path_pre = 'error_bottlecover_pre.txt'
path_gt = 'error_bottlecover_gt.txt'

# 加载数据
try:
    data_pre = np.loadtxt(path_pre)
    data_gt = np.loadtxt(path_gt)
    print("数据加载成功")
except Exception as e:
    print("数据加载失败:", e)

# 检查数据形状
print("预测数据形状:", data_pre.shape)
print("真实数据形状:", data_gt.shape)

# 确保数据维度相同
assert data_pre.shape == data_gt.shape, "预测值和真实值的维度不一致"

# 散点图比较
plt.figure(figsize=(12, 6))
plt.scatter(data_gt[:, 0], data_gt[:, 1], c='blue', label='真实值', alpha=0.5)
plt.scatter(data_pre[:, 0], data_pre[:, 1], c='red', label='预测值', alpha=0.5)
plt.title('预测值与真实值散点图比较')
plt.xlabel('X 坐标')
plt.ylabel('Y 坐标')
plt.legend()
plt.show()

# 线图比较
plt.figure(figsize=(12, 6))
plt.plot(data_gt[:, 0], label='真实值 X', color='blue')
plt.plot(data_pre[:, 0], label='预测值 X', color='red', linestyle='dashed')
plt.title('预测值与真实值线图比较')
plt.xlabel('样本索引')
plt.ylabel('X 坐标')
plt.legend()
plt.show()

# 可以为其他维度（Y坐标和Z坐标等）重复上面的步骤。
