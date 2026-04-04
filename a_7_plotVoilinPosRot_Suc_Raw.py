import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 文件路径映射
error_files_translation = {
    "peach": '/home/sen/Documents/InHand_pose/IMG_DATA_LS/IMG_DATA_PEACH_241227/Results/error_peach_241227_translation.txt',
}

error_files_rotation = {
    "peach": '/home/sen/Documents/InHand_pose/IMG_DATA_LS/IMG_DATA_PEACH_241227/Results/error_peach_241227_rotation.txt',
}

# 准备列表收集所有数据和标签
all_translation_errors = []
all_rotation_errors = []
objects_translation = []
objects_rotation = []

# 读取 translation 数据
for object_name, filepath in error_files_translation.items():
    translation_errors = np.loadtxt(filepath)
    all_translation_errors.extend(translation_errors)
    objects_translation.extend([object_name] * len(translation_errors))

# 读取 rotation 数据
for object_name, filepath in error_files_rotation.items():
    rotation_errors = np.loadtxt(filepath)
    rotation_errors = np.rad2deg(rotation_errors)  # 弧度转度数
    all_rotation_errors.extend(rotation_errors)
    objects_rotation.extend([object_name] * len(rotation_errors))

# 计算平均误差 
translation_mean_error = np.mean(all_translation_errors) 
rotation_mean_error = np.mean(all_rotation_errors)  
# 打印平均误差
print(f"Translation Mean Error: {translation_mean_error:.2f} mm") 
print(f"Rotation Mean Error: {rotation_mean_error:.2f} degrees")

# 创建 DataFrame
df_translation = pd.DataFrame({
    'Error': all_translation_errors,
    'Object': objects_translation,
    'Type': ['Translation'] * len(all_translation_errors)
})

df_rotation = pd.DataFrame({
    'Error': all_rotation_errors,
    'Object': objects_rotation,
    'Type': ['Rotation'] * len(all_rotation_errors)
})

# 合并 DataFrame
df = pd.concat([df_translation, df_rotation])

# 创建双 y 轴的图形
fig, ax1 = plt.subplots(figsize=(12, 8))

# 绘制平移误差的小提琴图
sns.violinplot(x="Object", y="Error", hue="Type", data=df[df['Type'] == 'Translation'], ax=ax1, inner="quartile", palette="Reds")
ax1.set_ylabel('Translation Error (mm)')

# 绘制旋转误差的小提琴图
ax2 = ax1.twinx()
sns.violinplot(x="Object", y="Error", hue="Type", data=df[df['Type'] == 'Rotation'], ax=ax2, inner="quartile", palette="Blues")
ax2.set_ylabel('Rotation Error (degree)')

# 设置图标题和属性
plt.title('Pose Estimation Error Without Filtering')
ax1.set_xlabel('Object')

# 去除重复图例
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1[:1] + h2[:1], ['Translation', 'Rotation'], loc='upper right')

# 保存和展示图像
plt.tight_layout()
plt.savefig('pose_estimation_error_without_filtering_' + datetime.now().strftime("%m%d-%H%M%S") + '.png', dpi=300)
plt.show()
