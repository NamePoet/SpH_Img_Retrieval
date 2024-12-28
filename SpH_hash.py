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


# 初始化随机中心
def random_center(data, n_bits):
    """
    随机初始化球面中心
    """
    N, D = data.shape
    centers = np.zeros((n_bits, D))
    for i in range(n_bits):
        samples = data[np.random.choice(N, 5, replace=False)]
        centers[i, :] = np.mean(samples, axis=0)
    return centers


# 统计计算函数
def compute_statistics(data, centers):
    """
    计算 o_i, o_ij, 半径, 平均值和标准差
    """
    N, _ = data.shape
    n_bits = centers.shape[0]

    dist = dist_mat(centers, data)
    sorted_dist = np.sort(dist, axis=1)
    radii = sorted_dist[:, N // 2]  # 平衡分区所需的半径

    dist_flags = (dist <= radii[:, np.newaxis]).astype(float)

    O1 = np.sum(dist_flags, axis=1)
    O2 = np.dot(dist_flags, dist_flags.T)

    avg = np.sum(np.abs(O2[np.triu_indices(n_bits, 1)] - N / 4)) / (n_bits * (n_bits - 1) / 2)
    avg2 = np.mean(O2[np.triu_indices(n_bits, 1)])
    stddev = np.sqrt(np.mean((O2[np.triu_indices(n_bits, 1)] - avg2) ** 2))

    return O1, O2, radii, avg, stddev


# 训练球面哈希
def spherical_hashing(data, n_bits):
    """
    训练球面哈希
    """
    N, D = data.shape
    centers = random_center(data, n_bits)

    iter_count = 0
    while True:
        O1, O2, radii, avg, stddev = compute_statistics(data, centers)

        forces = np.zeros_like(centers)
        for i in range(n_bits - 1):
            for j in range(i + 1, n_bits):
                force = 0.5 * (O2[i, j] - N / 4) / (N / 4) * (centers[i] - centers[j])
                forces[i] += force / n_bits
                forces[j] -= force / n_bits

        centers += forces

        if avg <= 0.1 * N / 4 and stddev <= 0.15 * N / 4:
            break
        if iter_count >= 100:
            print(f"Converged after 100 iterations with avg={avg}, stddev={stddev}")
            break

        iter_count += 1

    print(f"Training completed in {iter_count} iterations")
    return centers, radii


# 二值化数据
def compute_binary_codes(data, centers, radii):
    """
    使用球面中心和半径计算二值码
    """
    d_data = dist_mat(data, centers)  # 计算数据到球面中心的距离
    b_data = np.zeros_like(d_data, dtype=np.int32)  # 确保类型为整数
    b_data[d_data <= radii] = 1  # 根据半径生成二值化结果
    return compactbit(b_data)


# 加载和分割数据
data_file = "E:\\Pic_Process_2\\data.npz"
data = np.load(data_file)
features = data['arr_0']  # 图像特征，形状为 (16000, 768)

db_features = features[1000:16000]  # 数据库特征
n_bits = 64  # 二值码长度

# 训练球面哈希
centers, radii = spherical_hashing(db_features, n_bits)

# 计算二值码
db_train_features = compute_binary_codes(db_features, centers, radii)

# # 输出前 5 个二值化特征
# print("训练后的前 5 个二值化特征:")
# for i in range(5):
#     print(db_train_features[i])


# 解压缩二值化特征为逐位的 0/1
def decompressbit(compact_bits, n_bits):
    """
    将压缩的二值化特征展开为逐位的 0/1 表示
    """
    n_samples = compact_bits.shape[0]
    binary_features = np.unpackbits(compact_bits, axis=1)  # 解压到位级别
    return binary_features[:, :n_bits]  # 只保留有效位数

# 输出二值化特征及其原始特征值
print("训练后的前 5 个特征（原始特征值 + 二值化特征）：")
for i in range(5):
    original_feature = db_features[i]  # 原始特征值
    compact_feature = db_train_features[i]  # 压缩后的二值特征
    binary_feature = decompressbit(compact_feature[np.newaxis, :], n_bits)[0]  # 解压缩得到逐位 0/1

    # print(f"\n原始特征值[{i}]:")
    # print(original_feature)

    print(f"\n压缩后的二值化特征[{i}]:")
    print(compact_feature)

    print(f"\n逐位二值化特征[{i}]:")
    print(binary_feature)





# 计算 Hamming 距离
def compute_hamming_distance(query, database):
    """
    计算查询特征与数据库特征之间的 Hamming 距离
    query: 查询图像的二值化特征 (1D numpy array, shape: [n_bits])
    database: 数据库的二值化特征 (2D numpy array, shape: [num_samples, n_bits])
    返回距离值数组
    """
    # 按位异或计算差异，再统计 1 的数量
    distances = np.sum(query ^ database, axis=1)  # Hamming 距离
    return distances

# 按 Hamming 距离降序排列
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
        distances[query_index] = -1  # 标记为最小值，避免出现在排序结果中

    sorted_indices = np.argsort(-distances)  # 按距离降序排列
    return original_indices[sorted_indices], distances[sorted_indices]


# 解压二值化特征
db_binary_features = decompressbit(db_train_features, n_bits)  # 数据库解压后的二值特征

# 模拟查询
query_feature = db_binary_features[0]  # 假设使用数据库的第一个图像作为查询图像
original_indices = np.arange(len(db_binary_features))  # 数据库的图像索引

# 检索
sorted_indices, sorted_distances = retrieve_by_hamming(query_feature, db_binary_features, original_indices, query_index=0)

# 输出前 10 个检索结果
print("\n检索结果（前 10 个）：")
for i in range(10):
    print(f"索引: {sorted_indices[i]}, Hamming 距离: {sorted_distances[i]}")




# 每个图像有38类标签，标签以二值向量的形式存储（例如，长度为38的向量，1表示存在该标签，0表示不存在）
def compute_relevance(query_index, retrieved_indices, labels):
    """
    计算查询图像与检索结果之间的相关性。
    query_index: 查询图像在数据库中的索引
    retrieved_indices: 检索结果的图像索引
    labels: 数据库中每个图像的标签矩阵（每行是一个图像的标签）
    返回相关性向量 (与检索结果中每个图像的相关性 0 或 1)
    """
    query_labels = labels[query_index]
    relevance = np.zeros(len(retrieved_indices))
    for i, idx in enumerate(retrieved_indices):
        # 计算标签相交，若有交集则为相关图像
        if np.any(query_labels & labels[idx]):  # 检查标签的交集
            relevance[i] = 1
    return relevance


def compute_precision_at_k(retrieved_indices, relevance, K):
    """
    计算 precision@K
    """
    return np.sum(relevance[:K]) / K

def compute_recall_at_k(retrieved_indices, relevance, K, total_relevant):
    """
    计算 recall@K
    """
    return np.sum(relevance[:K]) / total_relevant

def compute_map(query_indices, retrieved_indices, labels, K):
    """
    计算 mAP
    """
    N_q = len(query_indices)
    total_map = 0
    for query_index in query_indices:
        relevance = compute_relevance(query_index, retrieved_indices[query_index], labels)
        relevant_count = np.sum(relevance)
        total_map += np.sum([compute_precision_at_k(retrieved_indices[query_index], relevance, k) * relevance[k-1]
                             for k in range(1, K+1)]) / relevant_count
    return total_map / N_q

def compute_recall(query_indices, retrieved_indices, labels, K):
    """
    计算 recall@K
    """
    N_q = len(query_indices)
    total_recall = 0
    for query_index in query_indices:
        relevance = compute_relevance(query_index, retrieved_indices[query_index], labels)
        total_relevant = np.sum(relevance)
        total_recall += compute_recall_at_k(retrieved_indices[query_index], relevance, K, total_relevant)
    return total_recall / N_q

def compute_precision(query_indices, retrieved_indices, labels, K):
    """
    计算 precision@K
    """
    N_q = len(query_indices)
    total_precision = 0
    for query_index in query_indices:
        relevance = compute_relevance(query_index, retrieved_indices[query_index], labels)
        total_precision += compute_precision_at_k(retrieved_indices[query_index], relevance, K)
    return total_precision / N_q



# 假设我们有一个标签矩阵，每个图像有38个标签
# 例如，labels[i] 表示第i个图像的标签，长度为38的二进制向量（1表示标签存在，0表示不存在）
labels = np.random.randint(2, size=(len(db_binary_features), 38))  # 示例标签，真实情况应加载实际标签数据

# 模拟查询
query_feature = db_binary_features[0]  # 假设使用数据库的第一个图像作为查询图像
original_indices = np.arange(len(db_binary_features))  # 数据库的图像索引

# 检索
sorted_indices, sorted_distances = retrieve_by_hamming(query_feature, db_binary_features, original_indices, query_index=0)

# 计算性能指标
query_indices = [0]  # 假设查询的是第0个图像
K = 10  # 计算前10个检索结果
mAP = compute_map(query_indices, sorted_indices, labels, K)
recall_at_k = compute_recall(query_indices, sorted_indices, labels, K)
precision_at_k = compute_precision(query_indices, sorted_indices, labels, K)

# 输出检索结果和性能指标

print(query_indices)
print("\n检索结果（前 10 个）：")
for i in range(10):
    print(f"索引: {sorted_indices[i]}, Hamming 距离: {sorted_distances[i]}")

print(f"\nmAP: {mAP}")
print(f"recall@{K}: {recall_at_k}")
print(f"precision@{K}: {precision_at_k}")













