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
paths = data['arr_2']     # 图像路径，形状为 (16000,)
labels = data['arr_1']      # 图像标签，形状为 (16000, 38)


# n_bits = 64  # 二值码长度

# 将数据分为查询集和数据库集 (特征0 路径2 标签1)
search_features = features[:1000]  # 前1000查询图像
db_features = features[1000:16000]     # 后15000数据库图像

search_paths = paths[:1000]
db_paths = paths[1000:16000]

search_labels = labels[:1000]      # 前1000查询图像
db_labels = labels[1000:16000]         # 后15000数据库图像

##########################################################################
# # 训练球面哈希
# centers, radii = spherical_hashing(db_features, n_bits)
#
# # 计算二值码
# db_train_features = compute_binary_codes(db_features, centers, radii)
#########################################################################

# 解压缩二值化特征为逐位的 0/1
def decompressbit(compact_bits, n_bits):
    """
    将压缩的二值化特征展开为逐位的 0/1 表示
    """
    n_samples = compact_bits.shape[0]
    binary_features = np.unpackbits(compact_bits, axis=1)  # 解压到位级别
    return binary_features[:, :n_bits]  # 只保留有效位数

#################################################################
# # 输出二值化特征及其原始特征值
# print("训练后的前 5 个特征（原始特征值 + 二值化特征）：")
# for i in range(5):
#     original_feature = db_features[i]  # 原始特征值
#     compact_feature = db_train_features[i]  # 压缩后的二值特征
#     binary_feature = decompressbit(compact_feature[np.newaxis, :], n_bits)[0]  # 解压缩得到逐位 0/1
#
#     # print(f"\n原始特征值[{i}]:")
#     # print(original_feature)
#
#     print(f"\n压缩后的二值化特征[{i}]:")
#     print(compact_feature)
#
#     print(f"\n逐位二值化特征[{i}]:")
#     print(binary_feature)
##################################################################

def compute_hamming_distance(query, database):
    """
    计算查询图像与数据库中所有图像的 Hamming 距离。
    query: 查询图像的二值化特征 (1D numpy array, shape: [n_bits])
    database: 数据库的二值化特征 (2D numpy array, shape: [num_samples, n_bits])
    返回距离值数组
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
        distances[query_index] = np.inf  # 标记为最大值，避免出现在排序结果中

    sorted_indices = np.argsort(distances)  # 按距离升序排列
    return original_indices[sorted_indices], distances[sorted_indices]


###################################################################################
# 解压二值化特征
# db_binary_features = decompressbit(db_train_features, n_bits)  # 数据库解压后的二值特征
#
#
# # 模拟查询
# query_feature = db_binary_features[0]  # 假设使用数据库的第一个图像作为查询图像
# original_indices = np.arange(len(db_binary_features))  # 数据库的图像索引
#
# # 检索
# sorted_indices, sorted_distances = retrieve_by_hamming(query_feature, db_binary_features, original_indices, query_index=0)
#
# # 输出前 10 个检索结果
#
# print("\n检索结果（前 10 个）：")
# for i in range(10):
#     print(f"索引: {sorted_indices[i]}, Hamming 距离: {sorted_distances[i]}")

##################################################################################

####这一步以上完全正确####

# 计算图像相关度
def compute_relevance(query_idx, sorted_indices, labels, k=10):
    """
    计算查询图像与数据库图像的相关度
    - 如果检索结果和查询图像具有相同的标签，则认为是相关图像
    """
    relevant_labels = labels[query_idx]  # 获取查询图像的标签
    relevant_count = 0  # 相关图像数量

    # 遍历前 K 个检索结果
    for i in range(k):
        retrieved_idx = sorted_indices[i]
        retrieved_labels = labels[retrieved_idx]

        # 检查标签是否相同
        if np.array_equal(relevant_labels, retrieved_labels):
            relevant_count += 1

    return relevant_count

####################################################################################
# 测试 compute_relevance 函数
def test_compute_relevance():
    # 模拟标签数据
    labels = np.array([
        [1, 0, 0],  # 图像 0 的标签
        [1, 0, 0],  # 图像 1 的标签
        [0, 1, 0],  # 图像 2 的标签
        [0, 1, 0],  # 图像 3 的标签
        [0, 0, 1],  # 图像 4 的标签
    ])

    # 模拟排序后的索引
    sorted_indices = np.array([0, 1, 2, 3, 4])  # 假设检索结果按顺序排列
    query_idx = 0  # 查询图像索引为 0

    # 计算相关度
    relevant_count = compute_relevance(query_idx, sorted_indices, labels, k=3)
    print(f"查询图像与前 3 个检索结果的相关度数量: {relevant_count}")

# test_compute_relevance()
#####################################################################################


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
    # 按维度逐位比较，检查是否存在至少一个维度为1
    relevance_matrix = (top_k_labels * query_label)  # 每个位置同时为1时，结果为1，否则为0
    relevance = np.any(relevance_matrix == 1, axis=1)  # 逐行判断是否至少有一个维度相关（正匹配）

    # Step 4: 计算 Precision 和 Recall
    retrieved = np.array([top_k_indices])  # 检索的前 k 个图像索引
    relevance_matrix_full = np.zeros((retrieved.shape[0], db_labels.shape[0]))  # 确保相关性矩阵的维度正确
    relevance_matrix_full[:, top_k_indices] = relevance  # 填充相关性矩阵

    precision = precision_at_k(retrieved, relevance_matrix_full, k)
    recall = recall_at_k(retrieved, relevance_matrix_full, k)
    mAP = mean_average_precision(retrieved, relevance_matrix_full, k_max=k)

    return precision, recall, mAP


#########################################################################
### 主程序
# 设置比特长度进行测试
bit_lengths = [32]       # 32,64,128

# 固定随机种子以保证结果可复现
np.random.seed(42)

# 随机选取查询图像的数量
n_query = 10

# 参数设置
k = 30  # 前 k 个检索结果

for n_bits in bit_lengths:
    print(f"\nTesting with {n_bits} bits:")

    # 训练球面哈希
    centers, radii = spherical_hashing(db_features, n_bits)

    # 计算数据库的二值化特征
    db_train_features = compute_binary_codes(db_features, centers, radii)
    db_binary_features = decompressbit(db_train_features, n_bits)  # 解压特征
    db_labels_binary = db_labels  # 数据库的标签

    # 从待查询图像中随机选取 10 张作为查询
    query_indices = np.random.choice(range(search_features.shape[0]), n_query, replace=False)

    # 初始化存储结果的列表
    precision_list = []
    recall_list = []
    mAP_list = []

    for query_idx in query_indices:
        # 查询图像的特征和标签
        query_feature = search_features[query_idx:query_idx + 1]  # 取第 query_idx 张查询图像
        query_label = search_labels[query_idx]

        # 对查询图像进行二值化
        query_binary = decompressbit(compute_binary_codes(query_feature, centers, radii), n_bits)[0]

        # 检索计算 Precision、Recall 和 mAP
        precision, recall, mAP = calculate_precision_recall_mAP(query_binary, db_binary_features, query_label, db_labels_binary, k)

        # 保存每张查询的结果
        precision_list.append(precision)
        recall_list.append(recall)
        mAP_list.append(mAP)

    # 计算随机选取的查询图像的平均 Precision、Recall 和 mAP
    mean_precision = np.mean(precision_list, axis=0)
    mean_recall = np.mean(recall_list, axis=0)
    mean_mAP = np.mean(mAP_list)

    # 输出结果
    print(f"\n随机选取的 {n_query} 张查询图像的平均 mAP 值: {mean_mAP:.4f}")
    print(f"随机选取的 {n_query} 张查询图像的平均 Precision 值 (前 {k} 个检索结果): {mean_precision}")
    print(f"随机选取的 {n_query} 张查询图像的平均 Recall 值 (前 {k} 个检索结果): {mean_recall}")

    # 检索前 10 个结果（以第一张查询图像为例）
    example_query_binary = decompressbit(compute_binary_codes(search_features[query_indices[0]:query_indices[0] + 1],
                                                              centers, radii), n_bits)[0]
    original_indices = np.arange(len(db_binary_features))
    sorted_indices, sorted_distances = retrieve_by_hamming(example_query_binary, db_binary_features, original_indices,
                                                           query_index=query_indices[0])

    # print(f"\n检索结果（前 10 个） for query index {query_indices[0]}:")
    # for i in range(10):
    #     print(f"索引: {sorted_indices[i]}, Hamming 距离: {sorted_distances[i]}")

#######################################################################################






