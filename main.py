import numpy as np

# 计算紧凑的二值码
def compactbit(b):
    """
    将二进制数组压缩成字节格式
    """
    n_samples, n_bits = b.shape
    n_words = (n_bits + 7) // 8  # 每 8 位一个字节
    cb = np.zeros((n_samples, n_words), dtype=np.uint8)

    for j in range(n_bits):
        w = j // 8  # 确定字节位置
        bit_pos = j % 8  # 位偏移
        cb[:, w] |= (b[:, j].astype(np.uint8) << bit_pos)  # 确保 b[:, j] 是整数
    return cb


# 计算欧氏距离
def dist_mat(P1, P2):
    """
    计算 P1 和 P2 行向量之间的欧氏距离
    """
    P1 = P1.astype(np.float64)
    P2 = P2.astype(np.float64)
    X1 = np.sum(P1 ** 2, axis=1, keepdims=True)
    X2 = np.sum(P2 ** 2, axis=1, keepdims=True)
    R = np.dot(P1, P2.T)
    D = np.sqrt(X1 + X2.T - 2 * R)
    return np.real(D)


# 解压缩二值化特征为逐位的 0/1
def decompressbit(compact_bits, n_bits):
    """
    将压缩的二值化特征展开为逐位的 0/1 表示
    """
    n_samples = compact_bits.shape[0]
    binary_features = np.unpackbits(compact_bits, axis=1)  # 解压到位级别
    return binary_features[:, :n_bits]  # 只保留有效位数


# 计算 Hamming 距离
def compute_hamming_distance(query, database):
    """
    计算查询图像与数据库中所有图像的 Hamming 距离。
    """
    return np.sum(query != database, axis=1)  # 按位异或


# 按 Hamming 距离升序排列
def retrieve_by_hamming(query, database, original_indices, query_index=None):
    """
    根据 Hamming 距离降序排列，同时忽略与查询自身的比较
    query: 查询图像的二值化特征 (1D numpy array)
    database: 数据库的二值化特征 (2D numpy array)
    original_indices: 数据库中图像的索引
    query_index: 查询图像在数据库中的索引（可选）
    返回排序后的索引和对应的距离
    """
    distances = compute_hamming_distance(query, database).astype(np.float32)

    # 忽略与自身的比较
    if query_index is not None:
        distances[query_index] = np.inf  # 标记为最小值，避免出现在排序结果中

    sorted_indices = np.argsort(distances)  # 按距离降序排列
    return original_indices[sorted_indices], distances[sorted_indices]


def precision_at_k(retrieved, relevance, k):
    """
    计算 precision@k
    retrieved: 检索的图像索引 (形状: [num_queries, k])
    relevance: 相关性矩阵 (形状: [num_queries, num_db_images])
    k: 计算 precision@k 的k值
    """
    precision = 0.0
    for i in range(retrieved.shape[0]):
        rel_at_k = relevance[i, retrieved[i, :k]]  # 获取第i个查询图像前k个检索结果的相关性
        precision += np.sum(rel_at_k) / k  # 累加前k个检索结果的精度
    return precision / len(retrieved)  # 平均精度


def recall_at_k(retrieved, relevance, k):
    """
    计算 recall@k
    retrieved: 检索的图像索引 (形状: [num_queries, k])
    relevance: 相关性矩阵 (形状: [num_queries, num_db_images])
    k: 计算 recall@k 的k值
    """
    recall = 0.0
    for i in range(retrieved.shape[0]):
        rel_at_k = relevance[i, retrieved[i, :k]]  # 获取第i个查询图像前k个检索结果的相关性
        recall += np.sum(rel_at_k) / np.sum(relevance[i])  # 累加召回率
    return recall / len(retrieved)  # 平均召回率


def mean_average_precision(retrieved, relevance, k_max=100):
    """
    计算 mAP
    retrieved: 检索的图像索引 (形状: [num_queries, num_db_images])
    relevance: 相关性矩阵 (形状: [num_queries, num_db_images])
    k_max: 最大k值（用于计算precision@k）
    """
    map_score = 0.0
    for i in range(retrieved.shape[0]):
        ap = 0.0
        relevant = 0
        for k in range(1, k_max + 1):
            if relevance[i, retrieved[i, k - 1]] == 1:  # 如果第k个检索结果相关
                relevant += 1
                ap += relevant / k  # 累加精度
        map_score += ap / np.sum(relevance[i])  # 计算该查询的平均精度（AP）
    return map_score / len(retrieved)  # 平均所有查询图像的平均精度


def calculate_precision_recall_mAP(query_binary, db_binary, query_label, db_labels, k):
    """
    计算 Precision, Recall 和 mAP。
    """
    # Step 1: 计算 Hamming 距离并升序排序
    hamming_distances = compute_hamming_distance(query_binary, db_binary)
    sorted_indices = np.argsort(hamming_distances)  # 升序排序，返回索引
    top_k_indices = sorted_indices[:k]  # 取前 k 个索引

    # Step 2: 提取前 k 个图像的标签
    top_k_labels = db_labels[top_k_indices]

    # Step 3: 判断相关性
    relevance = np.any(top_k_labels == query_label, axis=1)  # 任意一列标签相同则相关 (True/False)

    # Step 4: 计算 Precision 和 Recall
    retrieved = np.array([top_k_indices])  # 检索的前 k 个图像索引
    relevance_matrix = np.zeros((retrieved.shape[0], db_labels.shape[0]))  # 确保相关性矩阵的维度正确
    relevance_matrix[:, top_k_indices] = relevance  # 填充相关性矩阵

    precision = precision_at_k(retrieved, relevance_matrix, k)
    recall = recall_at_k(retrieved, relevance_matrix, k)
    mAP = mean_average_precision(retrieved, relevance_matrix, k_max=k)

    return precision, recall, mAP



# 测试计算函数
def test_calculate_precision_recall_mAP():
    # 假设查询图像和数据库的标签是已知的
    query_binary = np.random.randint(0, 2, size=64)  # 随机生成查询二值化特征
    db_binary = np.random.randint(0, 2, size=(16000, 64))  # 随机生成数据库二值化特征
    query_label = np.array([1, 0, 0])  # 假设查询图像的标签
    db_labels = np.random.randint(0, 2, size=(16000, 3))  # 随机生成数据库标签
    k = 10  # 取前10个检索结果

    precision, recall, mAP = calculate_precision_recall_mAP(query_binary, db_binary, query_label, db_labels, k)

    print(f"Precision at {k}: {precision:.4f}")
    print(f"Recall at {k}: {recall:.4f}")
    print(f"mAP: {mAP:.4f}")

# 调用测试函数
test_calculate_precision_recall_mAP()
