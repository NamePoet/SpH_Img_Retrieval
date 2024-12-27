import numpy as np
import time

def load_data(data_path):
    """
    加载数据。
    """
    data = np.load(data_path)
    features = data['arr_0']
    paths = data['arr_1']
    labels = data['arr_2']
    return features, paths, labels

def calculate_storage_consumption(num_bits, num_images):
    """
    计算存储消耗。
    """
    return num_bits * num_images / 8 / 1024  # KB

def calculate_avg_retrieval_time(query_binary_hash, db_binary_hash):
    """
    计算平均检索时间。
    """
    start_time = time.time()
    for query in query_binary_hash:
        _ = np.sum(query != db_binary_hash, axis=1)
    end_time = time.time()
    return (end_time - start_time) / len(query_binary_hash)
