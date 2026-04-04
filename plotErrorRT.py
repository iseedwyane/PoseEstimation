import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
rotation_errors = np.loadtxt('error_bottlecover_rotation.txt')
translation_errors = np.loadtxt('error_bottlecover_translation.txt')

# 设置画布
plt.figure(figsize=(12, 6))

# 画小提琴图 - 旋转误差
plt.subplot(1, 2, 1)
sns.violinplot(data=rotation_errors, inner="quartile")
plt.title('Rotation Error Distribution')
plt.ylabel('Rotation Error (rad)')

# 画小提琴图 - 平移误差
plt.subplot(1, 2, 2)
sns.violinplot(data=translation_errors, inner="quartile")
plt.title('Translation Error Distribution')
plt.ylabel('Translation Error (units)')

plt.tight_layout()
plt.show()

# 加载数据
rotation_errors = np.loadtxt('error_baseball_rotation.txt')
translation_errors = np.loadtxt('error_baseball_translation.txt')

# 设置画布
plt.figure(figsize=(12, 6))

# 画小提琴图 - 旋转误差
plt.subplot(1, 2, 1)
sns.violinplot(data=rotation_errors, inner="quartile")
plt.title('Rotation Error Distribution')
plt.ylabel('Rotation Error (rad)')

# 画小提琴图 - 平移误差
plt.subplot(1, 2, 2)
sns.violinplot(data=translation_errors, inner="quartile")
plt.title('Translation Error Distribution')
plt.ylabel('Translation Error (units)')

plt.tight_layout()
plt.show()


# 加载数据
rotation_errors = np.loadtxt('error_strawberry_rotation.txt')
translation_errors = np.loadtxt('error_strawberry_translation.txt')

# 设置画布
plt.figure(figsize=(12, 6))

# 画小提琴图 - 旋转误差
plt.subplot(1, 2, 1)
sns.violinplot(data=rotation_errors, inner="quartile")
plt.title('Rotation Error Distribution')
plt.ylabel('Rotation Error (rad)')

# 画小提琴图 - 平移误差
plt.subplot(1, 2, 2)
sns.violinplot(data=translation_errors, inner="quartile")
plt.title('Translation Error Distribution')
plt.ylabel('Translation Error (units)')

plt.tight_layout()
plt.show()

# 加载数据
rotation_errors = np.loadtxt('error_can_rotation.txt')
translation_errors = np.loadtxt('error_can_translation.txt')

# 设置画布
plt.figure(figsize=(12, 6))

# 画小提琴图 - 旋转误差
plt.subplot(1, 2, 1)
sns.violinplot(data=rotation_errors, inner="quartile")
plt.title('Rotation Error Distribution')
plt.ylabel('Rotation Error (rad)')

# 画小提琴图 - 平移误差
plt.subplot(1, 2, 2)
sns.violinplot(data=translation_errors, inner="quartile")
plt.title('Translation Error Distribution')
plt.ylabel('Translation Error (units)')

plt.tight_layout()
plt.show()



# 加载数据
rotation_errors = np.loadtxt('error_cube_rotation.txt')
translation_errors = np.loadtxt('error_cube_translation.txt')

# 设置画布
plt.figure(figsize=(12, 6))

# 画小提琴图 - 旋转误差
plt.subplot(1, 2, 1)
sns.violinplot(data=rotation_errors, inner="quartile")
plt.title('Rotation Error Distribution')
plt.ylabel('Rotation Error (rad)')

# 画小提琴图 - 平移误差
plt.subplot(1, 2, 2)
sns.violinplot(data=translation_errors, inner="quartile")
plt.title('Translation Error Distribution')
plt.ylabel('Translation Error (units)')

plt.tight_layout()
plt.show()
