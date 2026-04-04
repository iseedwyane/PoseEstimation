import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression

# 错误类型和文件名映射
error_files = {
    "masterball": 'error_masterball_0907',
    "bottlecover": 'error_cover_0912',
    "peach": 'error_peach_0912',
}

# 定义维度的标题
titles = ['x', 'y', 'z', 'Rx', 'Ry', 'Rz']

# 定义每个图的横轴坐标范围
x_axis_limits = {
    'x': (-20, 20),
    'y': (-10, 10),
    'z': (-10, 10),
    'Rx': (-5, 5),
    'Ry': (-0.5, 0.5),
    'Rz': (-5, 5)
}

# 设置标准差阈值
std_threshold = 100

# 指定散点图和回归线的RGB颜色
scatter_color = (0.50, 0.5, 0.8)  # RGB颜色
line_color = (0.5, 0.16, 0.16)  # RGB颜色

# 创建一个 nx6 的子图布局
n = len(error_files)
fig, axes = plt.subplots(n, 6, figsize=(18, 3 * n))  # n行6列，图的总宽度18英寸，高度3英寸每行

# 循环读取每种类型的数据并绘图
for idx, (object_name, base_filepath) in enumerate(error_files.items()):
    prediction_file = base_filepath + '_pre.txt'
    ground_truth_file = base_filepath + '_gt.txt'

    # 加载数据
    prediction = np.loadtxt(prediction_file)
    ground_truth = np.loadtxt(ground_truth_file)

    # 绘制每个维度的预测值vs真实值
    for i in range(6):
        # 计算预测值和真实值之间的差异
        errors = prediction[:, i] - ground_truth[:, i]
        mean_error = np.mean(errors)
        std_error = np.std(errors)

        # 过滤掉超过标准差阈值的点
        inliers = np.abs(errors - mean_error) <= std_threshold * std_error

        filtered_ground_truth = ground_truth[inliers, i]
        filtered_prediction = prediction[inliers, i]

        axes[idx, i].scatter(filtered_ground_truth, filtered_prediction, alpha=0.5, color=scatter_color)

        # 回归模型
        reg = LinearRegression().fit(filtered_ground_truth.reshape(-1, 1), filtered_prediction)
        x_range = np.linspace(filtered_ground_truth.min(), filtered_ground_truth.max(), 100)
        y_range = reg.predict(x_range.reshape(-1, 1))

        # 画出回归线
        #axes[idx, i].plot(x_range, y_range, color=line_color, label='Regression Line')

        # 设置横轴坐标范围
        #axes[idx, 4].set_xlim(x_axis_limits[titles[4]])
        #axes[idx, 4].set_ylim(x_axis_limits[titles[4]])

        # 设置标题和标签
        axes[idx, i].set_title(f'{object_name} - {titles[i]}')
        axes[idx, i].set_xlabel('GT')
        if i == 0:
            axes[idx, i].set_ylabel('Pre')
        axes[idx, i].legend()

plt.tight_layout()
plt.savefig('pose_estimation_comparison_' + datetime.now().strftime("%m%d-%H%M%S") + '.png', dpi=300)  # 指定保存路径和DPI
plt.show()