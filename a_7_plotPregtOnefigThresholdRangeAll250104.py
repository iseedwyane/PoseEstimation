import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from datetime import datetime

# 错误类型和文件名映射
error_files = {
    "bottlecover": '/home/sen/Documents/InHand_pose/error_cover_0912',

}

# 定义维度的标题
titles = ['x', 'y', 'z', 'Rx', 'Ry', 'Rz']

# 自定义散点图和回归线的RGB颜色
scatter_color = (85 / 255, 31 / 255, 51 / 255)

# 创建一个 1x6 的子图布局
fig, axes = plt.subplots(1, 6, figsize=(18, 4))

# 循环读取每种类型的数据并绘图
for idx, (object_name, base_filepath) in enumerate(error_files.items()):
    prediction_file = base_filepath + '_pre.txt'
    ground_truth_file = base_filepath + '_gt.txt'

    # 加载数据
    y_out = np.loadtxt(prediction_file)
    y_gt = np.loadtxt(ground_truth_file)

    # 对 x、y、z、Rx、Ry 数据取负号
    y_out[:, [0, 1, 2, 3, 4]] = -y_out[:, [0, 1, 2, 3, 4]]
    y_gt[:, [0, 1, 2, 3, 4]] = -y_gt[:, [0, 1, 2, 3, 4]]

    # 计算 MAE 和 R2
    mae = metrics.mean_absolute_error(y_gt, y_out, multioutput='raw_values')
    r2 = metrics.r2_score(y_gt, y_out, multioutput='raw_values')
    print(f"Object: {object_name}")
    print("Best MAE: ", [round(i, 3) for i in mae])
    print("Best R2 score: ", [round(i, 3) for i in r2])

    # 绘制每个维度的散点图
    for i in range(6):
        ax = axes[i]
        ax.scatter(y_gt[:, i], y_out[:, i], s=5, alpha=0.5, color=scatter_color)
        ax.set_title(titles[i])
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Prediction")
        ax.grid(True)
        ax.set_aspect('equal', adjustable='box')  # 坐标轴比例相等

# 调整布局并保存图像 save to where
plt.tight_layout()
plt.savefig('pose_estimation_comparison_' + datetime.now().strftime("%m%d-%H%M%S") + '.png', dpi=300)
plt.show()
