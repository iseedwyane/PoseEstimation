import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression

# 加载数据
#prediction = np.loadtxt('error_masterball_0824_pre.txt')
#ground_truth = np.loadtxt('error_masterball_0824_gt.txt')

# 加载数据
prediction = np.loadtxt('/home/sen/Documents/InHand_pose/IMG_DATA_LS/IMG_DATA_PEACH_241227/Results/error_peach_241227_pre.txt')
ground_truth = np.loadtxt('/home/sen/Documents/InHand_pose/IMG_DATA_LS/IMG_DATA_PEACH_241227/Results/error_peach_241227_gt.txt')

# 定义维度的标题
titles = ['x(mm)', 'y(mm)', 'z(mm)', 'Rx(degree)', 'Ry(degree)', 'Rz(degree)']

# 创建一个 1x6 的子图布局
fig, axes = plt.subplots(1, 6, figsize=(18, 3))  # 1行6列，图的总宽度18英寸，高度3英寸

# 设置标准差阈值
std_threshold = 0.5
std_threshold = 20
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

    axes[i].scatter(filtered_ground_truth, filtered_prediction, alpha=0.5)

    # 回归模型
    reg = LinearRegression().fit(filtered_ground_truth.reshape(-1, 1), filtered_prediction)
    x_range = np.linspace(filtered_ground_truth.min(), filtered_ground_truth.max(), 100)
    y_range = reg.predict(x_range.reshape(-1, 1))

    # 画出回归线，颜色为RGB
    #axes[i].plot(x_range, y_range, color=(0.0, 0.5, 1.0), label='Regression Line')  # RGB颜色

    axes[i].set_title(f'{titles[i]}')
    axes[i].set_xlabel('Griund Truth')
    if i == 0:
        axes[i].set_ylabel('Pre')
    #axes[i].legend()

plt.tight_layout()
plt.savefig('pose_estimation_comparison_'+ datetime.now().strftime("%m%d-%H%M%S") +'.png', dpi=300)  # 指定保存路径和DPI
plt.show()
