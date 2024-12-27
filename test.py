import numpy as np
import time

# 1.准备图像数据
# 数据文件路径
data_file = "E:\\Pic_Process_2\\data.npz"

# 加载数据
data = np.load(data_file)

# 提取特征、路径和标签
features = data['arr_0']  # 图像特征，形状为 (16000, 768)
labels = data['arr_1']     # 图像路径，形状为 (16000,)
paths = data['arr_2']    # 图像标签，形状为 (16000, 38)

"""
# 打印数据结构信息
# print("数据加载成功！")
print(f"特征 shape: {features.shape}, dtype: {features.dtype}")
print(f"路径 shape: {paths.shape}, dtype: {paths.dtype}")
print(f"标签 shape: {labels.shape}, dtype: {labels.dtype}")

# 打印样例数据
print("\n示例数据:")
print(f"第一个图像特征向量（前10个值）：{features[0][:10]}")
print(f"第一个图像路径：{paths[0]}")
print(f"第一个图像标签：{labels[0]}")  # 标签向量
"""

# 2.球面哈希函数，通过将图像的特征映射到一个高维的随机平面来计算哈希值。
def spherical_hash(features, num_bits):
    # 随机生成超平面
    random_planes = np.random.randn(num_bits, features.shape[1])

    # 计算投影并二值化
    projections = np.dot(features, random_planes.T)
    binary_hash = (projections > 0).astype(int)  # 将投影值二值化
    return binary_hash, random_planes

# 3.计算Hamming距离，衡量查询图像与数据库图像的相似度
def hamming_distance(hash1, hash2):
    return np.sum(hash1 != hash2, axis=1)

# 4.检索算法，使用计算出的 Hamming 距离对图像进行降序排序。对于每个查询图像，找到与之最不相似的k个图像。
# 冒泡排序，按照降序排列 Hamming 距离
# 快速排序函数（降序排列）
def quick_sort_descending(distances, indices):
    if len(distances) <= 1:
        return distances, indices
    pivot = distances[len(distances) // 2]
    left_distances, left_indices = [], []
    right_distances, right_indices = [], []
    middle_distances, middle_indices = [], []

    for i in range(len(distances)):
        if distances[i] > pivot:  # 比 pivot 大的放左边
            left_distances.append(distances[i])
            left_indices.append(indices[i])
        elif distances[i] < pivot:  # 比 pivot 小的放右边
            right_distances.append(distances[i])
            right_indices.append(indices[i])
        else:  # 等于 pivot 的放中间
            middle_distances.append(distances[i])
            middle_indices.append(indices[i])

    # 递归排序
    left_distances, left_indices = quick_sort_descending(left_distances, left_indices)
    right_distances, right_indices = quick_sort_descending(right_distances, right_indices)

    # 合并结果
    return left_distances + middle_distances + right_distances, left_indices + middle_indices + right_indices


def retrieve_images(query_hash, db_hash, K=5):
    # 计算每个查询图像与数据库图像的 Hamming 距离
    distances = hamming_distance(query_hash, db_hash)

    # 创建索引数组
    indices = np.arange(len(distances))

    # 使用快速排序按降序排列 Hamming 距离
    sorted_distances, sorted_indices = quick_sort_descending(distances, indices)

    # 打印一些调试信息来查看 Hamming 距离是否正确排序
    print(f"Query Hamming distances: {sorted_distances[:10]}")  # 打印前10个距离
    print(f"Sorted indices for the query: {sorted_indices[:10]}")  # 打印排序后的前10个索引

    return sorted_indices[:K]  # 返回前 K 个最不相关的图像索引

# 5.评估指标（mAP，Recall@K，Precision@K），根据检索结果计算检索的性能指标。
# 5.1 mAP (Mean Average Precision) 对于每个查询图像，计算其所有相关图像的精确度，并计算平均精度。
def calculate_map(query_labels, db_labels, sorted_indices):
    """
    # 计算 mAP
    ap = 0
    for i in range(len(query_labels)):
        relevant = (db_labels[sorted_indices] == query_labels[i])  # 相关标签
        ap += np.mean(relevant)  # 平均精度
    return ap / len(query_labels)
    """
    num_relevant = 0  # 累计相关图像数量
    precisions = []  # 存储每个相关结果的精度

    for i, idx in enumerate(sorted_indices):
        if np.array_equal(db_labels[idx], query_labels):  # 判断当前检索结果是否相关
            num_relevant += 1
            precision = num_relevant / (i + 1)  # 计算精度
            precisions.append(precision)  # 保存当前精度

    # 如果存在相关结果，返回 mAP，否则返回 0
    return np.mean(precisions) if precisions else 0

# 5.2 Recall@K 和 Precision@K  这两个指标衡量前 K 个检索结果中包含相关图像的比例。
"""
    计算 Recall@k 和 Precision@K
    query_labels: 查询图像的标签
    db_labels: 数据库中所有图像的标签
    sorted_indices: 按照检索结果排序后的图像索引
    K: 返回的前 K 个检索结果
"""
def recall_at_k(query_labels, db_labels, sorted_indices, K=5):   # 检索结果中相关的图像数 / 所有相关的图像数
    # 相关图像的索引在 sorted_indices 的末尾
    relevant = (db_labels[sorted_indices[-K:]] == query_labels)  # 判断前 K 个检索结果是否相关
    total_relevant = np.sum(db_labels == query_labels)  # 查询图像的总相关图像数
    recall = np.sum(relevant) / total_relevant if total_relevant > 0 else 0  # 计算 Recall@K

    # 打印每个查询的相关图像比例
    print(f"Total relevant for query: {total_relevant}, Relevant in top K: {np.sum(relevant)}")

    return recall

def precision_at_k(query_labels, db_labels, sorted_indices, K=5):  # 检索结果中相关的图像数 / k
    # 检索结果的最后 K 个图像是否相关
    relevant = (db_labels[sorted_indices[-K:]] == query_labels)
    precision = np.sum(relevant) / K
    return precision

# 6.主程序

# 将数据分为查询集和数据库集
query_features = features[:1000]  # 前1000查询图像
db_features = features[1000:]     # 后15000数据库图像
query_labels = labels[:1000]      # 前1000查询图像
db_labels = labels[1000:]         # 后15000数据库图像

# 用于测试检索性能的比特数：32, 64, 128
num_bits_list = [32, 64, 128]

# 遍历每个比特数，进行检索性能评估
for num_bits in num_bits_list:
    print(f"Evaluating for {num_bits}-bit hash...")

    # 对数据库图像进行哈希编码
    db_hash, random_planes = spherical_hash(db_features, num_bits)

    # 计算数据库特征存储消耗（计算数据库哈希特征的总字节数，输出以KB为单位）
    # 每个哈希特征是 num_bits 位，数据库有 len(db_features) 个图像
    db_storage = len(db_features) * num_bits / 8  # 存储消耗（字节）
    print(f"Database storage size: {db_storage / 1024:.2f} KB")  # 转换为 KB 输出


    # 对查询图像进行哈希编码
    query_hash, _ = spherical_hash(query_features, num_bits)

    # 计算并输出检索性能
    mAP = 0
    recall_list = []
    precision_list = []

    # 记录检索时间
    start_time = time.time()

    for i in range(len(query_features)):
        sorted_indices = retrieve_images(query_hash[i], db_hash)

        # 计算 mAP
        mAP += calculate_map(query_labels[i], db_labels, sorted_indices)

        # 计算 Recall@K 和 Precision@K
        recall_list.append(recall_at_k(query_labels[i], db_labels, sorted_indices))
        precision_list.append(precision_at_k(query_labels[i], db_labels, sorted_indices))

    # 计算检索时间
    total_time = time.time() - start_time  # 总时间
    avg_retrieval_time = total_time / len(query_features)  # 每张查询图像的平均检索时间

    # 输出评估结果
    print(f"mAP: {mAP / len(query_features):.4f}")
    print(f"Recall@K: {np.mean(recall_list):.4f}")
    print(f"Precision@K: {np.mean(precision_list):.4f}")
    print(f"Average retrieval time per query: {avg_retrieval_time:.6f} seconds")













