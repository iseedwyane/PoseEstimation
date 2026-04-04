import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression

# 错误类型和文件名映射
error_files = {
    "masterball": 'error_masterball_0912',
    "bottlecover": 'error_cover_0912',
    "peach": 'error_peach_0912',
}

# 定义维度的标题
titles = ['x', 'y', 'z', 'Rx', 'Ry', 'Rz']

# 为每个物体和每个维度定义横轴和纵轴的坐标范围和刻度
axis_limits = {
    "masterball": {
        'x': {'xlim': (0, 180), 'ylim': (0, 180), 'xticks': np.arange(0, 181, 40), 'yticks': np.arange(0, 181, 40)},
        'y': {'xlim': (-25, 35), 'ylim': (-25, 35), 'xticks': np.arange(-25, 36, 10), 'yticks': np.arange(-25, 36, 10)},
        'z': {'xlim': (-30, 70), 'ylim': (-30, 70), 'xticks': np.arange(-30, 71, 20), 'yticks': np.arange(-30, 71, 20)},
        'Rx': {'xlim': (-70, 90), 'ylim': (-70, 90), 'xticks': np.arange(-70, 91, 25), 'yticks': np.arange(-70, 91, 25)},
        'Ry': {'xlim': (-60, 90), 'ylim': (-60, 90), 'xticks': np.arange(-60, 91, 30), 'yticks': np.arange(-60, 91, 30)},
        'Rz': {'xlim': (0, 360), 'ylim': (0, 360), 'xticks': np.arange(0, 361, 90), 'yticks': np.arange(0, 361, 90)},
    },
    "bottlecover": {
        'x': {'xlim': (20, 180), 'ylim': (20, 180), 'xticks': np.arange(20, 181, 40), 'yticks': np.arange(20, 181, 40)},
        'y': {'xlim': (-30, 20), 'ylim': (-30, 20), 'xticks': np.arange(-30, 21, 10), 'yticks': np.arange(-30, 21, 10)},
        'z': {'xlim': (0, 90), 'ylim': (0, 90), 'xticks': np.arange(0, 91, 20), 'yticks': np.arange(0, 91, 20)},
        'Rx': {'xlim': (-75, 90), 'ylim': (-75, 90), 'xticks': np.arange(-75, 91, 50), 'yticks': np.arange(-75, 91, 50)},
        'Ry': {'xlim': (-30, 110), 'ylim': (-30, 110), 'xticks': np.arange(-30, 111, 40), 'yticks': np.arange(-30, 111, 40)},
        'Rz': {'xlim': (0, 360), 'ylim': (0, 360), 'xticks': np.arange(0, 361, 90), 'yticks': np.arange(0, 361, 90)},
    },
    "peach": {
        'x': {'xlim': (20, 180), 'ylim': (20, 180), 'xticks': np.arange(20, 181, 40), 'yticks': np.arange(20, 181, 40)},
        'y': {'xlim': (-10, 20), 'ylim': (-10, 20), 'xticks': np.arange(-10, 21, 10), 'yticks': np.arange(-10, 21, 10)},
        'z': {'xlim': (-10, 60), 'ylim': (-10, 60), 'xticks': np.arange(-10, 61, 20), 'yticks': np.arange(-10, 61, 20)},
        'Rx': {'xlim': (-75, 110), 'ylim': (-75, 110), 'xticks': np.arange(-75, 111, 50), 'yticks': np.arange(-75, 111, 50)},
        'Ry': {'xlim': (-60, 80), 'ylim': (-60, 80), 'xticks': np.arange(-60, 81, 30), 'yticks': np.arange(-60, 81, 30)},
        'Rz': {'xlim': (0, 360), 'ylim': (0, 360), 'xticks': np.arange(0, 361, 90), 'yticks': np.arange(0, 361, 90)},
    }
}

# 新的标准差阈值
std_threshold_xyz = 4  # 对 x, y, z 使用较宽松的阈值
std_threshold_rxrz = 3  # 对 Rx, Rz 使用中等宽松的阈值
std_threshold_ry = 2    # 对 Ry 使用稍微严格的阈值

# 最小保留数据比例
retain_ratio = 0.1  # 保留10%介于阈值之间的数据

# 自定义散点图和回归线的RGB颜色
scatter_color = (85/255, 31/255, 51/255)  # 自定义为图片中的颜色 #551F33
line_color = (0.8, 0.3, 0.3)  # 自定义颜色

# 创建一个 2x3 的子图布局，共6个子图
fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # 2行3列，图的总宽度18英寸，高度12英寸
axes = axes.ravel()  # 展平axes数组以便按顺序访问

# 定义要绘制的子图组合 (物体, 维度)
selected_plots = [
    
    ("bottlecover", "z"),  # 第二行第1列
    ("bottlecover", "y"),  # 第二行第2列
    ("bottlecover", "x"),  # 第二行第3列
    ("peach", "Rz"),  # 第一行第6列
    ("bottlecover", "Rx"),       # 第三行第4列
    ("peach", "Ry")   # 第一行第5列
]

# 循环绘制选定的子图
for ax_idx, (object_name, dimension) in enumerate(selected_plots):
    base_filepath = error_files[object_name]
    prediction_file = base_filepath + '_pre.txt'
    ground_truth_file = base_filepath + '_gt.txt'

    # 加载数据
    prediction = np.loadtxt(prediction_file)
    ground_truth = np.loadtxt(ground_truth_file)

    # 对 x、y、z、Rx、Ry 数据取负号
    prediction[:, [0, 1, 2, 3, 4]] = -prediction[:, [0, 1, 2, 3, 4]]
    ground_truth[:, [0, 1, 2, 3, 4]] = -ground_truth[:, [0, 1, 2, 3, 4]]

    # 获取当前维度的索引
    i = titles.index(dimension)
    
    # 计算预测值和真实值之间的差异
    errors = prediction[:, i] - ground_truth[:, i]
    mean_error = np.mean(errors)
    std_error = np.std(errors)

    # 根据维度名称选择不同的标准差阈值
    if dimension == 'Ry':
        inliers = np.abs(errors - mean_error) <= std_threshold_ry * std_error
    elif dimension in ['Rx', 'Rz']:
        inliers = np.abs(errors - mean_error) <= std_threshold_rxrz * std_error
    else:
        inliers = np.abs(errors - mean_error) <= std_threshold_xyz * std_error

    # 先应用较小的阈值去除异常数据
    initial_inliers = np.abs(errors - mean_error) <= 30 * std_error

    # 计算出误差位于 std_threshold_xyz=3 和 std_threshold_xyz=4 之间的数据
    extended_inliers = (np.abs(errors - mean_error) > 3 * std_error) & (np.abs(errors - mean_error) <= 4 * std_error)
    
    # 保留扩展区域中10%的数据
    extended_error_data = errors[extended_inliers]
    extended_ground_truth = ground_truth[extended_inliers, i]
    extended_prediction = prediction[extended_inliers, i]

    num_retain = max(1, int(len(extended_error_data) * retain_ratio))  # 确保至少保留1个数据
    if num_retain < len(extended_error_data):
        selected_indices = np.random.choice(len(extended_error_data), num_retain, replace=False)
        extended_error_data = extended_error_data[selected_indices]
        extended_ground_truth = extended_ground_truth[selected_indices]
        extended_prediction = extended_prediction[selected_indices]

    # 将剩下的初步筛选和额外保留的数据结合
    filtered_ground_truth = np.concatenate([ground_truth[initial_inliers, i], extended_ground_truth])
    filtered_prediction = np.concatenate([prediction[initial_inliers, i], extended_prediction])

    # 在当前子图中绘制散点图
    ax = axes[ax_idx]
    ax.scatter(filtered_ground_truth, filtered_prediction, s=5, alpha=0.5, color=scatter_color)

    # 设置每个维度的横纵坐标范围和刻度
    ax.set_xlim(axis_limits[object_name][dimension]['xlim'])
    ax.set_ylim(axis_limits[object_name][dimension]['ylim'])
    ax.set_xticks(axis_limits[object_name][dimension]['xticks'])
    ax.set_yticks(axis_limits[object_name][dimension]['yticks'])
    
    # 设置子图标题
    ax.set_title(f"{object_name} - {dimension}")
    
    # 设置每个子图的坐标轴比例为相等
    ax.set_aspect('equal', adjustable='box')

# 调整布局并保存图像
plt.tight_layout()
plt.savefig('selected_pose_estimation_comparison_2x3_' + datetime.now().strftime("%m%d-%H%M%S") + '.png', dpi=300)
plt.show()