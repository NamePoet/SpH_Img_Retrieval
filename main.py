import numpy as np
from hashing import train_hashing, compute_binary_hash
from evaluation import evaluate
from utils import calculate_storage_consumption, calculate_avg_retrieval_time, load_data

# 数据路径
data_path = r"E:\Pic_Process_2\data.npz"

if __name__ == "__main__":
    # 加载数据
    features, paths, labels = load_data(data_path)

    # 数据划分
    query_features = features[:1000]
    query_labels = labels[:1000]
    db_features = features[1000:16000]
    db_labels = labels[1000:16000]

    # 参数设置
    bit_lengths = [32, 64, 128]
    K_values = [10, 50, 100]

    # 实验开始
    for num_bits in bit_lengths:
        print(f"\nEvaluating for {num_bits}-bit hash...")

        # 训练球面哈希
        db_binary_hash, random_planes = train_hashing(db_features, num_bits)


        # 检查标签数据结构
        print(f"query_labels shape: {query_labels.shape}, dtype: {query_labels.dtype}")
        print(f"db_labels shape: {db_labels.shape}, dtype: {db_labels.dtype}")
        print(f"query_labels sample: {query_labels[0]}")

        # 检索性能评估
        mean_map, mean_recall, mean_precision = evaluate(
            query_features, query_labels,
            db_binary_hash, db_labels,
            random_planes, K_values
        )

        # 结果输出
        print(f"mAP: {mean_map}")
        for K in K_values:
            print(f"Recall@{K}: {mean_recall[K]:.4f}, Precision@{K}: {mean_precision[K]:.4f}")

        # 存储消耗与检索时间
        storage = calculate_storage_consumption(num_bits, len(db_features))
        avg_time = calculate_avg_retrieval_time(compute_binary_hash(query_features, random_planes), db_binary_hash)
        print(f"Storage consumption: {storage:.2f} KB")
        print(f"Average retrieval time: {avg_time:.6f} seconds")

