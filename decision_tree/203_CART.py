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
import pandas as pd
import numpy as np

from pprint import pprint
from collections import Counter
from dataset import load_watermelon_2_alpha

class CART(object):
    def __init__(self, depth=5) -> None:
        # self.shape = (X.shape[0], X.shape[1] + 1)
        self.tree = list()
        self.depth = 5
        super().__init__()
    
    def separate(self, X, y, parent_node, parent='root', depth=1):
        min_gini = 9999
        res_dict = parent_node.copy()

        if X.shape[0] == 0:
            # 若无数据输入
            return 1
        elif y.nunique() == 1:
            # 若多类型
            return 1

        for _, col in enumerate(X.columns):  # 对每个属性进行划分
            gini_index_a = gini_index(
                X, y, col
            )
            if gini_index_a < min_gini:
                min_gini = gini_index_a
                res_dict['gini'] = gini_index_a
                res_dict['a'] = col
                res_dict['parent'] = parent
        res_dict['depth'] += 1

        # 深度判断
        if res_dict['depth'] >= self.depth:
            return 1
        
        # 剪枝判断

        
        # 继续划分
        a_unique_values = df[res_dict['a']].unique()
        for a_value in a_unique_values:
            v_df = df.loc[df[res_dict['a']]==a_value]
            res_dict['data'] = list(v_df.index)
            res_dict['a_v'] = res_dict['a'] + '-' + a_value

            # update self.tree
            rv = self.separate(
                X=v_df.loc[:, [x for x in list(v_df.columns) if x != 'target']], 
                y=v_df['target'], 
                parent_node=res_dict,
                parent=res_dict['a'] + '-' + a_value
            )
            if rv == 1:
                self.tree.append({'a': res_dict['a'], 'parent': res_dict['parent'], 'counter': dict(Counter(v_df['target'])),
                                  'depth': res_dict['depth']})
    
    def fit(self, X, y, prune='pre'):
        """
        拟合CART
        """
        # init res_node
        init_node = {
            'a': 'all',
            'gini': 999,
            'a_v': 'root',
            'depth': 0,
            'parent': 'root',
        }
        self.separate(X, y, init_node)


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

        # 对缺失值进行处理，测试数据中，av = 0，则为缺失。
        # 将该样本按权重放入所有分支中
        
        dv_gini = gini(y_dv)
        gini_value += (per * dv_gini)
    return gini_value


def pre_prune_fit():
    pass

def after_prune_fit():
    pass


df = load_watermelon_2_alpha()
df.fillna('NULL', inplace=True)
X = df.loc[:, [x for x in list(df.columns) if x != 'target']]
y = df.target
print(df)

cart = CART()
cart.fit(X, y)
pprint(cart.tree)
