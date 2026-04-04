import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import os
import numpy as np
from scipy import interpolate

# 读取CSV文件
current_dir = os.getcwd()
print(current_dir)
filename = os.path.join(current_dir, 'closeloopResuts', 'demo2_final_result.csv')

# 读取文件并跳过第一行
data = pd.read_csv(filename, skiprows=1, header=None)

# 打印数据信息以便调试
print("数据形状:", data.shape)
print("列名:", data.columns.tolist())
print("前几行数据:")
print(data.head())

# 提取时间和数据列
time_seconds = data.iloc[:, 0]  # 第0列：时间数据（秒）
y1 = data.iloc[:, 1]  # ati (第1列)
y2 = data.iloc[:, 2]  # soft (第2列)
time_seconds_yaw = data.iloc[:, 4]  # yaw的时间数据（第4列）
yaw = data.iloc[:, 5]  # yaw (第5列)
refH = -0.15 * pd.Series([1] * len(data))  # 创建参考线

print("时间范围:", time_seconds.min(), "到", time_seconds.max(), "秒")
print("Yaw时间范围:", time_seconds_yaw.min(), "到", time_seconds_yaw.max(), "秒")
print("ATI范围:", y1.min(), "到", y1.max())
print("Soft范围:", y2.min(), "到", y2.max())
print("Yaw范围:", yaw.min(), "到", yaw.max())
print("参考线值: -0.15")

# 将yaw数据插值到与传感器数据相同的时间轴上
# 创建插值函数
yaw_interp_func = interpolate.interp1d(time_seconds_yaw, yaw, 
                                      bounds_error=False, 
                                      fill_value=(yaw.iloc[0], yaw.iloc[-1]))

# 在传感器数据的时间点上插值yaw数据
yaw_interpolated = yaw_interp_func(time_seconds)

# 设置合适的坐标轴范围
x_min = 0
x_max = 25.2

# 设置视频保存路径
output_video = 'demo2_final_result_animation.mp4'

# 创建图形和双y轴
fig, ax1 = plt.subplots(figsize=(12, 7))
ax2 = ax1.twinx()  # 创建第二个y轴

# 初始化图形 - 主要数据在左边y轴
line2, = ax1.plot([], [], color='#DC143C', label=r'Ground Truth', linewidth=2)
line3, = ax1.plot([], [], color='#483D8B', label=r'Predict value, $T_c$', linewidth=2)
line1, = ax1.plot([], [], color='#778899', label=r'Target Value', linestyle='--', linewidth=2)
# 初始化图形 - yaw数据和参考线在右边y轴
line4, = ax2.plot([], [], color='#FFD000', label=r'Yaw Angle', linewidth=2)

# 设置左边y轴范围
ax1.set_xlim(x_min, x_max)
ax1.set_ylim(-0.3, 0.2)
ax1.set_xlabel('Time (s)', fontsize=12)
ax1.set_ylabel('Torque (Nm)', fontsize=12, color='black')
ax1.tick_params(axis='y', labelcolor='black')

# 设置右边y轴范围（yaw数据）
ax2.set_ylim(-100, 720)
ax2.set_ylabel('Yaw (Deg)', fontsize=12, color='black')
ax2.tick_params(axis='y', labelcolor='black')

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

ax1.grid(True, alpha=0.7)

# 更新函数：每一帧更新图形
def update(frame):
    if frame > 0:
        current_time = time_seconds[:frame]
        line1.set_data(current_time, refH[:frame])  # 更新参考值
        line2.set_data(current_time, y1[:frame])    # 更新ATI数据
        line3.set_data(current_time, y2[:frame])    # 更新Soft数据
        line4.set_data(current_time, yaw_interpolated[:frame])   # 更新插值后的Yaw数据
        
    return line1, line2, line3, line4

# 创建动画
ani = animation.FuncAnimation(fig, update, frames=len(time_seconds), 
                             interval=50, blit=True, repeat=False)

# 保存动画为MP4文件
print("正在保存动画...")
writer = FFMpegWriter(fps=15, bitrate=5000)
ani.save(output_video, writer=writer, dpi=150)
print(f"动画已保存为: {output_video}")

# 先显示静态图检查数据
plt.figure(figsize=(12, 6))
ax1_static = plt.gca()
ax2_static = ax1_static.twinx()

ax1_static.plot(time_seconds, y1, color='#DC143C', label=r'ATI Sensor', linewidth=2)
ax1_static.plot(time_seconds, y2, color='#483D8B', label=r'Soft Sensor', linewidth=2)
ax1_static.plot(time_seconds, refH, color='#778899', label=r'Reference (-0.15)', linestyle='--', linewidth=2)
ax2_static.plot(time_seconds, yaw_interpolated, color='#FFD000', label=r'Yaw Angle (Interpolated)', linewidth=2)

ax1_static.set_xlabel('Time (s)', fontsize=12)
ax1_static.set_ylabel('Sensor Values', fontsize=12, color='black')
ax1_static.tick_params(axis='y', labelcolor='black')
ax1_static.set_ylim(-0.3, 0.2)

ax2_static.set_ylabel('Yaw Angle / Reference', fontsize=12, color='black')
ax2_static.tick_params(axis='y', labelcolor='black')
ax2_static.set_ylim(-100, 700)

# 合并图例
lines1, labels1 = ax1_static.get_legend_handles_labels()
lines2, labels2 = ax2_static.get_legend_handles_labels()
ax1_static.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

ax1_static.grid(True, alpha=0.7)
ax1_static.set_title('Sensor Data and Yaw Angle Measurement', fontsize=14)
plt.tight_layout()
plt.show()

# 然后再运行动画
plt.tight_layout()
plt.show()