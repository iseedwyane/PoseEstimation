import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 错误类型和文件名映射
# error_files_translation = {
#     "masterball": 'error_masterball_0912_translation.txt',
#     "bottlecover": 'error_cover_0912_translation.txt',
#     "peach": 'error_peach_0912_translation.txt',
# }

# error_files_rotation = {
#     "masterball": 'error_masterball_0912_rotation.txt',
#     "bottlecover": 'error_cover_0912_rotation.txt',
#     "peach": 'error_peach_0912_rotation.txt',
# }

error_files_translation = {
    "peach": '/home/sen/Documents/InHand_pose/IMG_DATA_LS/IMG_DATA_VASE_0110_231011/Results/error_pvase_250111_translation.txt',
}

error_files_rotation = {
    "peach": '/home/sen/Documents/InHand_pose/IMG_DATA_LS/IMG_DATA_VASE_0110_231011/Results/error_pvase_250111_rotation.txt',
}

# 准备空的列表来收集所有数据和标签
all_translation_errors = []
all_rotation_errors = []
objects_translation = []
objects_rotation = []

# 循环读取每种类型的 translation 数据
for object_name, filepath in error_files_translation.items():
    # 加载平移误差数据，并过滤大于 160 的数据点
    translation_errors = np.loadtxt(filepath)
    filtered_translation_errors = translation_errors[translation_errors <= 25]
    
    # 添加数据和对应的标签
    all_translation_errors.extend(filtered_translation_errors)
    objects_translation.extend([object_name] * len(filtered_translation_errors))

# 循环读取每种类型的 rotation 数据
for object_name, filepath in error_files_rotation.items():
    # 加载旋转误差数据，并过滤大于 160 的数据点
    rotation_errors = np.loadtxt(filepath)
    #rotation_errors=np.degrees(rotation_errors)
    filtered_rotation_errors = rotation_errors[rotation_errors <= 100]
    filtered_rotation_errors = np.rad2deg(filtered_rotation_errors)######rad2degrad2degrad2degrad2degrad2degrad2deg
    # 添加数据和对应的标签
    all_rotation_errors.extend(filtered_rotation_errors)
    objects_rotation.extend([object_name] * len(filtered_rotation_errors))

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

# 创建一个带有两个 y 轴的图形
fig, ax1 = plt.subplots(figsize=(12, 8))



# 右侧 y 轴：旋转误差
ax2 = ax1.twinx()
sns.violinplot(x="Object", y="Error", hue="Type", data=df[df['Type'] == 'Rotation'], ax=ax2, inner="quartile", palette="Blues")
ax2.set_ylabel('Rotation Error (degree)')
#ax2.set_ylim(0, 180)  # 根据数据调整 y 轴范围
# 左侧 y 轴：平移误差
sns.violinplot(x="Object", y="Error", hue="Type", data=df[df['Type'] == 'Translation'], ax=ax1, inner="quartile", palette="Reds")
ax1.set_ylabel('Translation Error (mm)')
#ax1.set_ylim(0, 25)  # 根据数据调整 y 轴范围

# 设置图标题和其他属性
plt.title('Pose Estimation Error')
ax1.set_xlabel('Object')

# 移除重复的图例
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1[:1] + h2[:1], ['Translation', 'Rotation'], loc='upper right')

# 保存图像
plt.tight_layout()
plt.savefig('pose_estimation_error_' + datetime.now().strftime("%m%d-%H%M%S") + '.png', dpi=300)
plt.show()