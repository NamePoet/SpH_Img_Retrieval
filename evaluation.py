import numpy as np
from sklearn.metrics import average_precision_score

def hamming_distance(query_hash, db_hash):
    """
    计算查询图像与数据库图像的Hamming距离。
    """
    return np.sum(query_hash != db_hash, axis=1)

def calculate_map(query_labels, db_labels, sorted_indices):
    """
    计算mAP（平均精度）。
    """
    relevant = np.sum(query_labels & db_labels, axis=1) > 0
    y_true = relevant[sorted_indices]
    y_score = np.linspace(1, 0, len(y_true))  # 模拟的检索分数
    return average_precision_score(y_true, y_score)

def calculate_recall_precision(query_labels, db_labels, sorted_indices, K):
    """
    计算recall@K 和 precision@K。
    """
    relevant = np.sum(query_labels & db_labels, axis=1) > 0
    y_true = relevant[sorted_indices[:K]]
    recall = np.sum(y_true) / np.sum(relevant)
    precision = np.sum(y_true) / K
    return recall, precision

def evaluate(query_features, query_labels, db_binary_hash, db_labels, random_planes, K_values):
    """
    对检索性能进行评估。
    """
    query_binary_hash = compute_binary_hash(query_features, random_planes)
    mAP_list = []
    recall_list = {k: [] for k in K_values}
    precision_list = {k: [] for k in K_values}

    for i in range(len(query_features)):
        distances = hamming_distance(query_binary_hash[i], db_binary_hash)
        sorted_indices = np.argsort(distances)

        # 计算mAP
        mAP = calculate_map(query_labels[i], db_labels, sorted_indices)
        mAP_list.append(mAP)

        # 计算recall@K 和 precision@K
        for K in K_values:
            recall, precision = calculate_recall_precision(query_labels[i], db_labels, sorted_indices, K)
            recall_list[K].append(recall)
            precision_list[K].append(precision)

    # 平均结果
    mean_map = np.mean(mAP_list)
    mean_recall = {k: np.mean(recall_list[k]) for k in K_values}
    mean_precision = {k: np.mean(precision_list[k]) for k in K_values}

    return mean_map, mean_recall, mean_precision
