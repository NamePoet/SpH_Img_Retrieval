import numpy as np
import sys
import os

# 数据文件路径

# 路径 特征 标签
np.set_printoptions(threshold=sys.maxsize)
arr_2 = np.load("E:\\Pic_Process_2\\data\\arr_2.npy", allow_pickle=True)
# print(arr_2[:5])
print(arr_2[0])
print(arr_2[999]) #第1000张
print(arr_2[1000])
print(arr_2[15999])

np.set_printoptions(threshold=sys.maxsize)
arr_0 = np.load("E:\\Pic_Process_2\\data\\arr_0.npy", allow_pickle=True)
# print(arr_0[:5])

np.set_printoptions(threshold=sys.maxsize)
arr_1 = np.load("E:\\Pic_Process_2\\data\\arr_1.npy", allow_pickle=True)
print(arr_1[0])
print(arr_1[999]) #第1000张
print(arr_1[1000])
print(arr_1[15999])