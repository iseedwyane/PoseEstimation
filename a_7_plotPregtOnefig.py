import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 加载数据
prediction = np.loadtxt('error_masterball_0907_pre.txt')
ground_truth = np.loadtxt('error_masterball_0907_gt.txt')

# 定义维度的标题
titles = ['x', 'y', 'z', 'Rx', 'Ry', 'Rz']

# 创建一个 1x6 的子图布局
fig, axes = plt.subplots(1, 6, figsize=(18, 3))  # 1行6列，图的总宽度18英寸，高度3英寸

# 绘制每个维度的预测值vs真实值
for i in range(6):
    axes[i].scatter(ground_truth[:, i], prediction[:, i], alpha=0.5)
    axes[i].set_title(f'{titles[i]}')
    axes[i].set_xlabel('GT')
    if i == 0:
        axes[i].set_ylabel('Pre')

plt.tight_layout()
plt.savefig('pose_estimation_comparison_'+ datetime.now().strftime("%m%d-%H%M%S") +'.png', dpi=300)  # 指定保存路径和DPI
plt.show()