import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_violin(data_paths, object_name):
    # 加载数据
    rotation_errors = np.loadtxt(data_paths['rotation'])
    translation_errors = np.loadtxt(data_paths['translation'])
    
    # 设置画布
    plt.figure(figsize=(12, 6))
    
    # 画小提琴图 - 旋转误差
    plt.subplot(1, 2, 1)
    sns.violinplot(data=rotation_errors, inner="quartile")
    plt.title(f'{object_name} Rotation Error Distribution')
    plt.ylabel('Rotation Error (rad)')
    
    # 画小提琴图 - 平移误差
    plt.subplot(1, 2, 2)
    sns.violinplot(data=translation_errors, inner="quartile")
    plt.title(f'{object_name} Translation Error Distribution')
    plt.ylabel('Translation Error (units)')
    
    plt.tight_layout()
    plt.show()

# 错误类型和文件名映射
error_files = {
    "bottlecover": {"rotation": 'error_bottlecover_rotation.txt', "translation": 'error_bottlecover_translation.txt'},
    "baseball": {"rotation": 'error_baseball_rotation.txt', "translation": 'error_baseball_translation.txt'},
    "cube": {"rotation": 'error_cube_rotation.txt', "translation": 'error_cube_translation.txt'},
    "strawberry": {"rotation": 'error_strawberry_rotation.txt', "translation": 'error_strawberry_translation.txt'},
    "can": {"rotation": 'error_can_rotation.txt', "translation": 'error_can_translation.txt'}
}

# 绘制每个物体的误差图
for object_name, paths in error_files.items():
    plot_violin(paths, object_name)

