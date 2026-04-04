import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
# 错误类型和文件名映射
error_files = {
    "masterball": 'error_masterball_0830_translation.txt',
    "bottlecover": 'error_bottle_0830_translation.txt',
    "peach": 'error_peach_0830_translation.txt',
}

# 准备空的列表来收集所有数据和标签
all_errors = []
objects = []

# 循环读取每种类型的数据
for object_name, filepath in error_files.items():
    # 加载平移误差数据，并过滤大于 45 的数据点
    translation_errors = np.loadtxt(filepath)
    filtered_errors = translation_errors[translation_errors <= 160]
    
    # 添加数据和对应的标签
    all_errors.extend(filtered_errors)
    objects.extend([object_name] * len(filtered_errors))

# 创建 DataFrame
df = pd.DataFrame({
    'Translation Error': all_errors,
    'Object': objects
})

# 绘制小提琴图
plt.figure(figsize=(10, 6))
sns.violinplot(x="Object", y="Translation Error", data=df, inner="quartile")
plt.title('Translation Error Distribution Across Different Objects')
plt.ylabel('Translation Error (units)')
plt.xlabel('Object')
plt.xticks(rotation=0)

# 保存图像
plt.tight_layout()
plt.savefig('translation_error_distribution_'+ datetime.now().strftime("%m%d-%H%M%S") +'.png', dpi=300)  # 指定保存路径和DPI
plt.show()

