import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 错误类型和文件名映射
error_files = {
    "bottlecover": {"rotation": 'error_bottlecover_rotation.txt', "translation": 'error_bottlecover_translation.txt'},
    "baseball": {"rotation": 'error_baseball_rotation.txt', "translation": 'error_baseball_translation.txt'},
    #"cube": {"rotation": 'error_cube_rotation.txt', "translation": 'error_cube_translation.txt'},
    "strawberry": {"rotation": 'error_strawberry_rotation.txt', "translation": 'error_strawberry_translation.txt'},
    #"can": {"rotation": 'error_can_rotation.txt', "translation": 'error_can_translation.txt'}
}

# 准备空的列表来收集所有数据和标签
all_errors = []
labels = []
error_types = []

# 循环读取每种类型的数据
for object_name, paths in error_files.items():
    # 加载旋转误差数据
    rotation_errors = np.loadtxt(paths['rotation'])
    all_errors.extend(rotation_errors)
    labels.extend([object_name] * len(rotation_errors))
    error_types.extend(['Rotation'] * len(rotation_errors))
    
    # 加载平移误差数据
    translation_errors = np.loadtxt(paths['translation'])
    all_errors.extend(translation_errors)
    labels.extend([object_name] * len(translation_errors))
    error_types.extend(['Translation'] * len(translation_errors))

# 创建 DataFrame
df = pd.DataFrame({
    'Error': all_errors,
    'Object': labels,
    'Error Type': error_types
})

# 绘制小提琴图
plt.figure(figsize=(14, 6))
sns.violinplot(x="Object", y="Error", hue="Error Type", data=df, split=True, inner="quartile")
plt.title('Error Distribution Across Different Objects and Error Types')
plt.ylabel('Error Magnitude')
plt.xticks(rotation=45)  # 如果标签名较长，可以旋转以便更好地显示
plt.legend(title='Error Type')
plt.tight_layout()
plt.show()
