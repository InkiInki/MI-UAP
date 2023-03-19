# coding: utf-8
"""
作者: 因吉
邮箱: inki.yinji@qq.com
创建: 2021 0805
修改: 2021 0806
"""


import numpy as np
from scipy.io import loadmat
from numba import jit


def get_iter(tr, tr_lab, te, te_lab):
    """
    获取单词迭代器
    :param  tr:     训练集
    :param  tr_lab: 训练集标签
    :param  te:     测试集
    :param  te_lab: 测试集标签
    :return 相应迭代器
    """
    yield tr, tr_lab, te, te_lab


def get_k_cv_idx(num_x, k=10, seed=None):
    """
    获取k次交叉验证的索引
    :param num_x:       数据集的大小
    :param k:           决定使用多少折的交叉验证
    :param seed:        随机种子
    :return:            训练集索引，测试集索引
    """
    # 随机初始化索引
    if seed is not None:
        np.random.seed(666)
    rand_idx = np.random.permutation(num_x)
    # 每一折的大小
    fold = int(np.floor(num_x / k))
    ret_tr_idx = []
    ret_te_idx = []
    for i in range(k):
        # 获取当前折的训练集索引
        tr_idx = rand_idx[0: i * fold].tolist()
        tr_idx.extend(rand_idx[(i + 1) * fold:])
        ret_tr_idx.append(tr_idx)
        # 添加当前折的测试集索引
        ret_te_idx.append(rand_idx[i * fold: (i + 1) * fold].tolist())
    return ret_tr_idx, ret_te_idx


def get_performance(type_performance):
    """
    获取分类性能度量
    :param type_performance: 分类性能度量指标
    :return: 分类性能度量函数
    """
    ret_per = {}
    for type_per in type_performance:
        if type_per == "acc":
            from sklearn.metrics import accuracy_score
            metric = accuracy_score
        else:
            from sklearn.metrics import f1_score
            metric = f1_score
        ret_per[type_per] = metric

    return ret_per


def print_progress_bar(idx, size):
    """
    打印进度条
    :param idx:    当前位置
    :param size:   总进度
    """
    print('\r' + '▇' * int(idx // (size / 50)) + str(np.ceil((idx + 1) * 100 / size)) + '%', end='')


def load_file(data_path):
    """
    载入.mat类型的多示例数据集
    :param data_path:  数据集的存储路径
    """
    return loadmat(data_path)['data']


@jit(nopython=True)
def dis_euclidean(arr1, arr2):
    """"""
    return np.sqrt(np.sum((arr1 - arr2)**2))


@jit(nopython=True)
def kernel_rbf(arr1, arr2=None, gamma=1):
    """"""
    if arr2 is None:
        return np.sqrt(np.sum(arr1**2))
    return np.exp(-gamma * dis_euclidean(arr1, arr2))
