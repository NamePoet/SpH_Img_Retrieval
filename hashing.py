import numpy as np

def generate_random_planes(dim, num_bits):
    """
    随机生成超平面，用于特征投影。
    :param dim: 输入特征维度
    :param num_bits: 二值哈希码的长度
    :return: 随机超平面矩阵
    """
    return np.random.randn(dim, num_bits)

def compute_binary_hash(features, random_planes):
    """
    使用随机超平面对特征进行二值化。
    :param features: 输入特征矩阵 (N, dim)
    :param random_planes: 随机超平面矩阵 (dim, num_bits)
    :return: 二值化特征矩阵 (N, num_bits)
    """
    projections = np.dot(features, random_planes)  # 特征投影
    binary_hash = (projections >= 0).astype(np.uint8)  # 二值化
    return binary_hash

def train_hashing(db_features, num_bits):
    """
    训练球面哈希函数，生成数据库的二值哈希特征。
    :param db_features: 数据库特征 (N, dim)
    :param num_bits: 哈希长度
    :return: 二值哈希特征矩阵 (N, num_bits), 随机超平面
    """
    random_planes = generate_random_planes(db_features.shape[1], num_bits)
    db_binary_hash = compute_binary_hash(db_features, random_planes)
    return db_binary_hash, random_planes
