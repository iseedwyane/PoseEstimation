import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import os
import numpy as np

# 读取CSV文件
current_dir = os.getcwd()
print(current_dir)
filename = os.path.join(current_dir, 'closeloopResuts', 'demo1_processed_data_final_mu.csv')

# 读取文件并跳过第一行
data = pd.read_csv(filename, skiprows=1, header=None)

# 打印数据信息以便调试
print("数据形状:", data.shape)
print("列名:", data.columns.tolist())
print("前几行数据:")
print(data.head())

# 提取时间和数据列
time_seconds = data.iloc[:, 0]  # 第0列：时间数据（秒）
y1 = data.iloc[:, 1]  # 第1列：Fx_N (夹持力)
y2 = data.iloc[:, 2]  # 第2列：NegFz_N (剪切力)
mu = data.iloc[:, 3]  # 第3列：μ数据
refH = 0.45 * pd.Series([1] * len(data))  # 创建参考线

print("时间范围:", time_seconds.min(), "到", time_seconds.max(), "秒")
print("Fx_N范围:", y1.min(), "到", y1.max(), "N")
print("NegFz_N范围:", y2.min(), "到", y2.max(), "N")
print("μ范围:", mu.min(), "到", mu.max())
print("参考线值: 0.45")

# 设置合适的坐标轴范围
x_min = time_seconds.min()
x_max = time_seconds.max()

# 设置视频保存路径
output_video = 'demo1_processed_data_final_A.mp4'

# 创建图形和双y轴
fig, ax1 = plt.subplots(figsize=(12, 7))
ax2 = ax1.twinx()  # 创建第二个y轴

# 初始化图形 - 力数据在左边y轴
line2, = ax1.plot([], [], color='#DC143C', label=r'Gripping Force, $F_g$ ', linewidth=2)
line3, = ax1.plot([], [], color='#483D8B', label=r'Shear Force, $F_{s,z}$', linewidth=2)

# 初始化图形 - μ数据和参考线在右边y轴
line4, = ax2.plot([], [], color='#FFD000', label=r'$\mu = F_{s,z} / F_g$', linewidth=2)
line1, = ax2.plot([], [], color='#778899', label=r'$Coeficient = 0.45$', linestyle='--', linewidth=2)

# 设置左边y轴范围（力数据）
ax1.set_xlim(x_min, x_max)
ax1.set_ylim(0, 10)
ax1.set_xlabel('Time (s)', fontsize=12)
ax1.set_ylabel('Force (N)', fontsize=12, color='black')
ax1.tick_params(axis='y', labelcolor='black')

# 设置右边y轴范围（系数参考线）
ax2.set_ylim(0, 1)  # 右边y轴范围0-1
ax2.set_ylabel('Coefficient $\mu$', fontsize=12, color='black')
ax2.tick_params(axis='y', labelcolor='black')

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

ax1.grid(True, alpha=0.7)
#ax1.set_title('Force and Friction Coefficient Measurement', fontsize=14)

# 更新函数：每一帧更新图形
def update(frame):
    if frame > 0:
        current_time = time_seconds[:frame]
        line1.set_data(current_time, refH[:frame])  # 更新参考值
        line2.set_data(current_time, y1[:frame])    # 更新夹持力
        line3.set_data(current_time, y2[:frame])    # 更新剪切力
        line4.set_data(current_time, mu[:frame])    # 更新μ数据
    return line1, line2, line3, line4  # 必须返回所有4条线！

# 创建动画
ani = animation.FuncAnimation(fig, update, frames=len(time_seconds), 
                             interval=50, blit=True, repeat=False)

# 保存动画为 MP4 文件
print("正在保存动画...")
writer = FFMpegWriter(fps=15, bitrate=5000)
ani.save(output_video, writer=writer, dpi=150)
print(f"动画已保存为: {output_video}")

# 先显示静态图检查数据
plt.figure(figsize=(12, 6))
ax1_static = plt.gca()
ax2_static = ax1_static.twinx()

ax1_static.plot(time_seconds, y1, color='#DC143C', label=r'$F_g$ (Gripping Force)', linewidth=2)
ax1_static.plot(time_seconds, y2, color='#483D8B', label=r'$F_{s,z}$ (Shear Force)', linewidth=2)
ax2_static.plot(time_seconds, mu, color='#FFD000', label=r'$\mu$ (Calculated)', linewidth=2)
ax2_static.plot(time_seconds, refH, color='#778899', label=r'$\mu_{ref} = 0.4$', linestyle='--', linewidth=2)

ax1_static.set_xlabel('Time (s)', fontsize=12)
ax1_static.set_ylabel('Force (N)', fontsize=12, color='black')
ax1_static.tick_params(axis='y', labelcolor='black')
ax1_static.set_ylim(0, 10)

ax2_static.set_ylabel('Friction Coefficient', fontsize=12, color='#778899')
ax2_static.tick_params(axis='y', labelcolor='#778899')
ax2_static.set_ylim(0, 1)

# 合并图例
lines1, labels1 = ax1_static.get_legend_handles_labels()
lines2, labels2 = ax2_static.get_legend_handles_labels()
ax1_static.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

ax1_static.grid(True, alpha=0.7)
ax1_static.set_title('Force and Friction Coefficient Measurement - Static Plot', fontsize=14)
plt.tight_layout()
plt.show()

# 然后再运行动画
plt.tight_layout()
plt.show()