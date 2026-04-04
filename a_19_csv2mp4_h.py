import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import os

# 读取CSV文件
current_dir = os.getcwd()
print(current_dir)
filename1 = os.path.join(current_dir, 'closeloopResuts', 'output_data0712-223945.csv')
data = pd.read_csv(filename1)
filename2 = os.path.join(current_dir, 'closeloopResuts', 'output_data0712-223945_re.csv')
data2 = pd.read_csv(filename2)

#print(data)

# 确保第一列是时间类型（如果它是字符串）
data.iloc[:, 0] = pd.to_datetime(data.iloc[:, 0])

# 提取时间和数据列
time = data.iloc[:, 0]  # 第一列作为时间数据
y1 = data2.iloc[:, 2]  # （预测值）
y2 = - data2.iloc[:, 13]   # （假设是实际值）
refH = 90 * pd.Series([1] * len(data))  # 创建参考高度列
#print(refH)

# 处理时间数据，转换为秒
initial_time = time.iloc[0]
time_seconds = (time - initial_time).dt.total_seconds()  # 转换为相对秒数

# 设置视频保存路径
output_video = 'output_video_h.mp4'
# 创建视频写入对象
writer = FFMpegWriter(fps=8)  # 每秒30帧
fig, ax = plt.subplots()
# 初始化图形
line1, = ax.plot([], [], color='#778899', label='Ref', linestyle='--')  # 参考值（红色虚线）
line2, = ax.plot([], [], color='#DC143C', label='Ground Truth')  # 实际值（黄色虚线）
line3, = ax.plot([], [], color='#483D8B', label='Predict Value')  # 预测值（蓝色实线）

ax.set_xlim(0, 27.5)  # 设置横轴范围
ax.set_ylim(0, 145)  # 设置纵轴范围
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Lift Height (mm)', fontsize=12)
ax.legend(loc='lower right')
ax.grid(True)

# 更新函数：每一帧更新图形
def update(frame):
    line1.set_data(time_seconds[:frame], refH[:frame])  # 更新参考值
    line2.set_data(time_seconds[:frame], y2[:frame])  # 更新实际值
    line3.set_data(time_seconds[:frame], y1[:frame])  # 更新预测值
    return line1, line2, line3  # 返回更新后的线对象

# 创建动画
ani = animation.FuncAnimation(fig, update, frames=len(time_seconds), blit=True)

# 保存动画为 MP4 文件
ani.save(output_video, writer=writer, dpi=300)

# 显示图形
plt.show()
