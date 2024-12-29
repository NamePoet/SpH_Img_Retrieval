# 计算 Precision@k
def precision_at_k(retrieved_indices, relevant_labels, k):
    """
    计算前 K 个检索结果的精度
    """
    # 确保索引不会越界
    retrieved_indices = np.clip(retrieved_indices, 0, len(relevant_labels) - 1)
    retrieved_labels = relevant_labels[retrieved_indices[:k]]  # 获取前 K 个检索结果的标签
    relevant_count = np.sum(retrieved_labels == 1)  # 查找相关标签的数量
    precision = relevant_count / k  # 计算精度
    return precision


# 计算 Recall@k
def recall_at_k(retrieved_indices, relevant_labels, k, total_relevant):
    """
    计算前 K 个检索结果的召回率
    """
    # 确保索引不会越界
    retrieved_indices = np.clip(retrieved_indices, 0, len(relevant_labels) - 1)
    retrieved_labels = relevant_labels[retrieved_indices[:k]]  # 获取前 K 个检索结果的标签
    relevant_count = np.sum(retrieved_labels == 1)  # 查找相关标签的数量
    recall = relevant_count / total_relevant  # 计算召回率
    return recall


# 计算 mAP
def mean_average_precision(query_indices, database_binary_features, n_bits, k=10):
    """
    计算检索的 mAP
    """
    average_precisions = []
    for query_idx in query_indices:
        query_feature = database_binary_features[query_idx]
        sorted_indices, _ = retrieve_by_hamming(query_feature, database_binary_features,
                                                np.arange(len(database_binary_features)), query_index=query_idx)
        relevant_labels = labels[query_idx]  # 获取查询图像的标签
        total_relevant = np.sum(relevant_labels)  # 相关标签的总数
        precision = precision_at_k(sorted_indices, relevant_labels, k)
        recall = recall_at_k(sorted_indices, relevant_labels, k, total_relevant)

        # 输出 Precision@k 和 Recall@k
        # print(f"Query {query_idx} - Precision@{k}: {precision:.4f}, Recall@{k}: {recall:.4f}")

        average_precisions.append(precision)

    mAP = np.mean(average_precisions)
    return mAP


# 设置不同的比特长度进行测试
bit_lengths = [32, 64, 128]

for n_bits in bit_lengths:
    print(f"\nTesting with {n_bits} bits:")

    # 训练球面哈希
    centers, radii = spherical_hashing(db_features, n_bits)

    # 计算二值码
    db_train_features = compute_binary_codes(db_features, centers, radii)

    # 解压二值化特征
    db_binary_features = decompressbit(db_train_features, n_bits)

    # 模拟查询
    query_feature = db_binary_features[0]  # 使用数据库的第一个图像作为查询图像
    original_indices = np.arange(len(db_binary_features))

    # 检索
    sorted_indices, sorted_distances = retrieve_by_hamming(query_feature, db_binary_features, original_indices,
                                                           query_index=0)

    # 输出前 10 个检索结果
    print("\n检索结果（前 10 个）：")
    for i in range(10):
        print(f"索引: {sorted_indices[i]}, Hamming 距离: {sorted_distances[i]}")

    # 计算 mAP
    query_indices = np.random.choice(len(db_binary_features), 100)  # 随机选择 100 个查询
    mAP = mean_average_precision(query_indices, db_binary_features, n_bits)
    print(f"Mean Average Precision (mAP) for {n_bits} bits: {mAP:.4f}")