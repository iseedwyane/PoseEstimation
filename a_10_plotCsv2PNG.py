import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import numpy as np
# 读取CSV文件
filename = np.loadtxt('aruco_data/aruco_id12_data_20250808_161303.csv', delimiter=',')#

#filename = 'aruco_data/aruco_id12_data_20250808_161303.csv' # baseball
#filename = 'output_data0813-203951.csv' 
# filename = 'output_data0712-231538.csv'
#filename = 'output_data0712-223945_refine.csv' # cube
data = pd.read_csv(filename)

numRows, numCols = data.shape

refH = [90] * numRows

# 显示第11行的数据
print(data.iloc[10, :])

# 以第一列做横轴数据-时间，第3列和第9列分别为纵轴数据，绘制图形
time = pd.to_datetime(data.iloc[:, 0]) # 第一列作为时间数据

# 将时间减去初始时间，并转换为秒
initial_time = time.iloc[0]
time_seconds = (time - initial_time).dt.total_seconds() # 转换为相对的秒数

y1 = data.iloc[:, 6] # 第三列数据
y2 = data.iloc[:, 12] # 第九列数据
y2 = np.rad2deg(y2) -200
plt.figure()

plt.plot(time_seconds, refH, color=[0.5, 0.54, 0.53], linewidth=2, linestyle='--', label='Ref Height')
plt.plot(time_seconds, y1, color=[0.10, 0.10, 0.44], linewidth=2, linestyle='-', label='Predict Height') # 绘制第3列数据，蓝色线
plt.plot(time_seconds, -y2, color=[1, 0.6, 0.07], linewidth=2, linestyle='-.', label='Ground Truth') # 绘制第9列数据，红色线

# 设置横纵轴的范围
plt.xlabel('Time (s)', fontsize=12) # 横轴标签，字体大小12
plt.ylabel('Height (mm)', fontsize=12) # 纵轴标签，字体大小12
# plt.title('Predicted Height vs Ground Truth over Time') # 图形标题
plt.legend(loc='lower right') # 显示图例
# plt.grid(True) # 显示网格

# 设置横纵轴的范围
plt.xlim([0, max(time_seconds)]) # 根据时间数据的范围设置横轴范围
plt.ylim([min(min(y1), min(-y2)), max(max(y1), max(-y2))]) # 根据数据范围设置纵轴范围

# 设置图形尺寸为8x6厘米，300dpi
plt.gcf().set_size_inches(8/2.54, 6/2.54) # 转换为英寸
plt.gca().tick_params(axis='both', which='major', labelsize=10) # 设置坐标轴刻度字体大小

plt.savefig('output_figure.png', dpi=300) # 以300dpi保存图形为PNG格式文件
plt.show()