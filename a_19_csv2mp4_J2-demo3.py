import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import os
import numpy as np

# 读取CSV文件
current_dir = os.getcwd()
print(current_dir)
filename = os.path.join(current_dir, 'closeloopResuts', 'demo3_fts_plot_data_demolightbulb.csv')

# 读取文件并跳过第一行
data = pd.read_csv(filename, skiprows=1, header=None)

# 打印数据信息以便调试
print("数据形状:", data.shape)
print("列名:", data.columns.tolist())
print("前几行数据:")
print(data.head())

# 提取时间和数据列
time_seconds = data.iloc[:, 0]  # 第0列：时间数据（秒）
y1 = data.iloc[:, 2]  # Fz
y2 = data.iloc[:, 3]  # Fy


print("时间范围:", time_seconds.min(), "到", time_seconds.max(), "秒")
print("GroundTruth范围:", y1.min(), "到", y1.max(), "mm")
print("Predict范围:", y2.min(), "到", y2.max(), "mm")

# 设置合适的坐标轴范围
x_min = time_seconds.min()
x_max = time_seconds.max()


# 设置视频保存路径
output_video = 'demo3_fts_plot_data_demolightbulb.mp4'

# 创建图形
fig, ax = plt.subplots(figsize=(12, 7))

# 初始化空线条

line2, = ax.plot([], [], color='#DC143C', label='$F_{s,z}$', linewidth=2)
line3, = ax.plot([], [], color='#483D8B', label='$F_{s,y}$', linewidth=2)

# 设置坐标轴范围
ax.set_xlim(x_min, x_max)
ax.set_ylim(-3, 5)
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Force (N)', fontsize=12)
ax.legend(loc='upper left')
ax.grid(True)

# 预计算数据点数量
total_frames = len(time_seconds)

# 更新函数：每一帧更新图形
def update(frame):
    # 只绘制到当前帧的数据，而不是每次都绘制所有数据
    #line1.set_data(time_seconds[:frame+1], refH[:frame+1])
    line2.set_data(time_seconds[:frame+1], y1[:frame+1])
    line3.set_data(time_seconds[:frame+1], y2[:frame+1])
    return line2, line3

# 创建动画
ani = animation.FuncAnimation(fig, update, frames=total_frames, 
                             interval=100, blit=True, repeat=False)

# 保存动画为 MP4 文件
print("正在保存动画...")
writer = FFMpegWriter(fps=10, bitrate=5000)  # 降低fps使动画更流畅
ani.save(output_video, writer=writer, dpi=150)
print(f"动画已保存为: {output_video}")

# 可选：同时绘制静态图进行检查
plt.figure(figsize=(12, 6))
plt.plot(time_seconds, y1, color='#DC143C', label='Ground Truth', linewidth=2)
plt.plot(time_seconds, y2, color='#483D8B', label='Predict Value', linewidth=2)

plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Height (mm)', fontsize=12)
plt.legend()
plt.grid(True)
plt.title('Height Comparison - Static Plot')
plt.tight_layout()
plt.savefig('static_plot_comparison.png')
plt.show()