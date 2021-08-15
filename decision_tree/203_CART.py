# -*- encoding: utf-8 -*-
'''
@File    :   203_CART.py
@Time    :   2021/08/02 17:17:10
@Author  :   qiujiayu 
@Version :   1.0
@Contact :   qiujy@highlander.com.cn
@Desc    :   CART demo
'''

# here put the import lib
from math import inf
import pandas as pd
import numpy as np

from pprint import pprint
from collections import Counter
from dataset import load_watermelon_3
# from sklearn.tree import DecisionTreeClassifier

from anytree import Node


# class RowSample(object):
#     def __init__(self, tid, weight, row_series, row_target) -> None:
#         self.tid = tid
#         self.weight = weight
#         self.row_series = row_series
#         self.row_target = row_target
#         super().__init__()


class CART(object):
    def __init__(self, depth=5) -> None:
        # self.shape = (X.shape[0], X.shape[1] + 1)
        self.tree = list()
        self.depth = 5
        super().__init__()
    
    def generate_tree(self, D, a, p_value):
        """生成树

        Args:
            D ([type]): [样本集合]
            a ([type]): [划分属性]
            p_value ([type]): [父节点取值]

        Returns:
            [type]: [description]
        """
        # init tree
        node = Node(p_value)

        # 判断 样本集合是否都是同一样本
        D_unique = D.drop_duplicates(keep=False)
        if D_unique.shape[0] == 0:
            # 样本全部相同，返回树
            node.target = D['好瓜'].value_counts().index[0]  # 返回标签频次最高对应的值
            return node
        
        # 判断 样本中标签是否完全相同
        if D['好瓜'].nunique() == 1:
            # 所有样本对应标签都一致，返回树
            node.target = D['好瓜'].values[0]
            return node

        # 计算样本中gini指数最小的属性，作为最优划分属性
        gini_index = float(inf)
        best_attr = 'x'

        for _, a in enumerate(list(D.columns)):  # 逐个属性进行计算gini指数
            pass

    
    def fit(self, X, y):
        """
        拟合CART
        """
        self.generate_tree(X, y, 'root')


def gini(y) -> float:
    """基尼指数，计算样本纯度

    基尼指数越小，则说明数据集纯度越高(样本中可分类的数据越少)。

    Args:
        y : [样本标签]

    Returns:
        float: [样本纯度对应的基尼指数]]
    """
    target_counter_dict = dict(Counter(y))
    target_size = len(y)

    pk_square_sum = 0
    for target_value in target_counter_dict.keys():
        pk = target_counter_dict[target_value] / target_size
        pk_square_sum += (pk ** 2)
    gini = 1 - pk_square_sum
    return gini


def gini_index(X, y, flag: str) -> float:
    """对属性a的基尼指数

    Args:
        X ([type]): [特征]
        y ([type]): [标签]
        flag (str): [属性a对应的列名]

    Returns:
        float: [属性a的基尼指数]
    """
    d_size = X.shape[0]  # 样本数量
    Av = list(set((X[flag])))  # 在属性a上，a对应的取值列表
    Av_size = len(Av)  # 在属性a上，a对应的取值数量
    Av_counter_dict = dict(Counter(X[flag]))  # 对Av进行统计

    gini_value = 0
    for _, av in enumerate(Av):  # 统计属性a上，每个取值对应的gini指数
        per = Av_counter_dict[av] / d_size
        a_col = X[flag]
        av_index = ~a_col.where(a_col == av).isna()  # 样本 在属性a上 取值 为 av 的样本索引

        X_dv = X.loc[av_index, :]
        y_dv = y.loc[av_index]
        
        dv_gini = gini(y_dv)
        gini_value += (per * dv_gini)
    return gini_value


def pre_prune_fit():
    pass

def after_prune_fit():
    pass


df = load_watermelon_3()
df.fillna('NULL', inplace=True)
X = df.loc[:, [x for x in list(df.columns) if x != '好瓜']]
y = df.target
print(df)

cart = CART()
cart.fit(X, y)
