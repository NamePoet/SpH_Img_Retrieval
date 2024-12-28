import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import os
import re

# 1.准备图像数据
# 数据文件路径
data_file = "E:\\Pic_Process_2\\data.npz"

# 加载数据
data = np.load(data_file)

# 提取特征、路径和标签
features = data['arr_0']  # 图像特征，形状为 (16000, 768)
paths = data['arr_2']     # 图像路径，形状为 (16000,)
labels = data['arr_1']      # 图像标签，形状为 (16000, 38)

# # 打印数据结构信息
# print(f"特征 shape: {features.shape}, dtype: {features.dtype}")
# print(f"路径 shape: {paths.shape}, dtype: {paths.dtype}")
# print(f"标签 shape: {labels.shape}, dtype: {labels.dtype}")


# 将数据分为查询集和数据库集 (特征0 路径2 标签1)
search_features = features[:1000]  # 前1000查询图像
db_features = features[1000:16000]     # 后15000数据库图像

search_paths = paths[:1000]
db_paths = paths[1000:16000]

search_labels = labels[:1000]      # 前1000查询图像
db_labels = labels[1000:16000]         # 后15000数据库图像

# print(search_labels[0])       # 1
# print(search_labels[999])     # 1000
#
# print(db_labels[0])          # 1001
# print(db_labels[14999])      # 16000

# 1. 定义球面哈希算法类
class SphericalHashing:
    def __init__(self, num_bits):
        self.num_bits = num_bits  # 哈希码长度
        self.centers = None       # 存储球面中心

    def train(self, data):
        """训练球面哈希函数"""
        # 使用PCA降维到与哈希码长度相同的维度
        pca = PCA(n_components=self.num_bits)
        reduced_data = pca.fit_transform(data)

        # 归一化每个样本，使其映射到单位球面上
        self.centers = normalize(reduced_data, axis=0)   # axis 1->0

    def hash(self, data):
        """将数据二值化"""
        # 将数据映射到球面中心
        projections = np.dot(data, self.centers.T)  # 点积计算

        # 生成二值哈希码（使用符号函数将正值置1，负值置0）
        binary_codes = (projections > 0).astype(int)
        return binary_codes


# 3. 初始化球面哈希算法
num_bits = 64  # 选择二值码长度
spherical_hashing = SphericalHashing(num_bits)

# 4. 训练球面哈希函数
spherical_hashing.train(db_features)

# 5. 对数据库特征进行二值化
db_train_features = spherical_hashing.hash(db_features)


# 6. 输出前5个二值化后的特征
# print("训练后的前5个二值化特征:")
# for i in range(5):
print("训练前特征：")
print(db_features[0])
print("训练后特征：")
print(db_train_features[0])