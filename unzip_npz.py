import numpy as np
import os

# 数据文件路径
data_file = "E:\\Pic_Process_2\\data.npz"

# 加载数据
data = np.load(data_file)

# 提取特征、路径和标签
features = data['arr_0']  # 图像特征，形状为 (16000, 768)
paths = data['arr_2']  # 图像路径，形状为 (16000,)
labels = data['arr_1']  # 图像标签，形状为 (16000, 38)

# 打印数据结构信息
print(f"特征 shape: {features.shape}, dtype: {features.dtype}")
print(f"路径 shape: {paths.shape}, dtype: {paths.dtype}")
print(f"标签 shape: {labels.shape}, dtype: {labels.dtype}")

# 输出前5个值的特征、标签和路径，并存入元组
print("\n前5个图像的信息（特征、标签、路径）：")
for i in range(5):
    feature = features[i]  # 获取第i幅图像的特征
    path = paths[i]  # 获取第i幅图像的路径
    label = labels[i]  # 获取第i幅图像的标签

    # 将图像的特征、路径和标签组合成一个元组
    image_info = (label, path)

    # 输出每幅图像的信息
    print(f"第{i + 1}幅图像的信息: {image_info}")
