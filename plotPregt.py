import numpy as np
import matplotlib.pyplot as plt

import os

# 获取当前工作目录
print("Current working directory:", os.getcwd())


# Use relative paths for files in the current directory
path_pre = 'error_bottlecover_pre.txt'
path_gt = 'error_bottlecover_gt.txt'

# Load data
try:
    data_pre = np.loadtxt(path_pre)
    data_gt = np.loadtxt(path_gt)
    print("Data loaded successfully")
except Exception as e:
    print("Failed to load data:", e)

# Check the shapes of the data arrays
print("Shape of the prediction data:", data_pre.shape)
print("Shape of the ground truth data:", data_gt.shape)

# Ensure that both data arrays have the same dimensions
assert data_pre.shape == data_gt.shape, "The dimensions of predictions and ground truth do not match"

# Plot a scatter plot for comparison
plt.figure(figsize=(12, 6))
plt.scatter(data_gt[:, 0], data_gt[:, 1], c='blue', label='Ground Truth', alpha=0.5)
plt.scatter(data_pre[:, 0], data_pre[:, 1], c='red', label='Predictions', alpha=0.5)
plt.title('Comparison of Predictions and Ground Truth (Scatter Plot)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.show()

# Plot line graphs for comparison
plt.figure(figsize=(12, 6))
plt.plot(data_gt[:, 0], label='Ground Truth X', color='blue')
plt.plot(data_pre[:, 0], label='Predictions X', color='red', linestyle='dashed')
plt.title('Comparison of Predictions and Ground Truth (Line Graph)')
plt.xlabel('Sample Index')
plt.ylabel('X Coordinate')
plt.legend()
plt.show()

# Additional plots for other dimensions (Y and Z coordinates) can be added similarly.
