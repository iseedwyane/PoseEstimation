import pandas as pd

# 读取已上传的 CSV 文件
file_path = './Transfered_output.csv'
df = pd.read_csv(file_path)

# 假设 temp 是一个包含多列的新数据
temp = [1, 2, 3, 4, 5]

# 将 temp 数据作为一行添加到 DataFrame 中
#df.loc[len(df)] = temp

# 保存更新后的 DataFrame 回到 CSV 文件
#df.to_csv(file_path, index=False)


print(f"Updated CSV file has been saved to {file_path}")

print(df)