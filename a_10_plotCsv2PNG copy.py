import numpy as np
import matplotlib.pyplot as plt

# 读取 CSV 文件

#data = np.loadtxt('demo-LiS/demo-exp-1-screwlightbulb-250519-173547/fts_cam7.csv', delimiter=',')# no movement, benchmark
data = np.loadtxt('aruco_data/aruco_id12_data_20250808_161303.csv', delimiter=',')#
#data = np.loadtxt('demo-LiS/demo-exp-1-screwlightbulb-250520-124752/fts_cam7.csv', delimiter=',')#
# 提取 fx (第0列), fz (第2列)
fx = data[:, 0]
fy = data[:, 1]
fz = data[:, 2]

# 计算 fx/fz (避免除以零)
ratio = np.divide(fz, fx, out=np.zeros_like(fz), where=fx!=0)

# 生成时间轴 (假设数据是等间隔采集的)
timestamps = np.arange(len(fx)) * 0.03  # 假设 30Hz 采样率 (0.03s/帧)



# 绘图
plt.figure(figsize=(12, 8))
plt.subplot(4, 1, 1)
plt.plot(timestamps, fx, 'r-', label='Fx')
plt.ylabel("Force (N)")
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(timestamps, fy, 'b-', label='Fy')
plt.ylabel("Force (N)")
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(timestamps, fz, 'b-', label='Fz')
plt.ylabel("Force (N)")
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(timestamps, ratio, 'g-', label='Fx/Fz')
plt.xlabel("Time (s)")
plt.ylabel("Ratio")
plt.legend()

plt.tight_layout()
plt.savefig("force_analysis.png")
plt.show()