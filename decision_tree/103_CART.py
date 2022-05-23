'''
@File    :   203_CART.py
@Time    :   2021/08/02 17:17:10
@Author  :   qiujiayu
@Version :   1.1
@Contact :   qiujy@highlander.com.cn
@Desc    :   CART demo 完成。
             考虑了缺失值、连续值的情况。
             且对模型进行后剪枝，预剪枝比较简单没有进行编写。
'''
from math import inf

import pandas as pd

from collections import Counter
import utils.dataset as dataset

from anytree import Node
from anytree import RenderTree
from anytree import find, findall


def get_rho(D, flag):
    """get rho by weights"""
    d_size = sum(D['权重'])
    a_not_null_size = sum(D.loc[D[flag]!='NULL']['权重'])
    return a_not_null_size / d_size


def get_pk(D, flag, target):
    """get pk"""
    a_not_null_size = sum(D.loc[D[flag]!='NULL']['权重'])
    if a_not_null_size == 0:
        print(D, flag)
    else:
        pass
    tk_size = sum(D.loc[D['好瓜']==target]['权重'])
    return tk_size / a_not_null_size


def get_rv(D, flag, av):
    """get rv"""
    a_not_null_size = sum(D.loc[D[flag]!='NULL']['权重'])
    av_size = sum(D.loc[D[flag]==av]['权重'])
    return av_size / a_not_null_size


def gini(D, flag) -> float:
    """基尼指数，计算样本纯度
    """
    y = D['好瓜']
    target_counter_dict = dict(Counter(y))

    pk_square_sum = 0
    for target_value in target_counter_dict.keys():
        pk = get_pk(D, flag, target_value)
        pk_square_sum += (pk ** 2)
    gini = 1 - pk_square_sum
    return gini


def gini_index(D, flag: str) -> float:
    """对属性a的基尼指数
    """
    if isinstance(D[flag].values[0], str):
        Av = [x for x in list(set((D[flag]))) if x != 'NULL']  # 在属性a上，a对应的取值列表

        # 计算在rho：在该属性上，非空取值权重占比
        rho = get_rho(D, flag)

        gini_value = 0
        for _, av in enumerate(Av):  # 统计属性a上，每个取值对应的gini指数
            rv = get_rv(D, flag, av)  # 计算rv，av为非空时，权重的占比
            a_col = D[flag]
            av_index = a_col == av

            dv_gini = gini(D.loc[av_index], flag)
            gini_value += (rv * dv_gini)
        return rho * gini_value, '离散'

    if isinstance(D[flag].values[0], float):
        return gini_index_float(D, flag)


def gini_index_float(D, flag: str) -> float:
    # 获取所有属性上的取值，并进行排序
    a = list(D[flag])
    a.sort()

    # 划分点集合
    T = []
    for i in range(len(a) - 1):
        T.append((a[i] + a[i+1]) / 2)

    # 按划分生成左右两个分类类型
    min_gini_index = float(inf)
    best_t = 'x'

    for t in T:
        left = D[flag][D[flag] <= t]
        right = D[flag][D[flag] > t]

        if (len(left) == 0) | (len(right) == 0):
            t_gini = float(inf)
        else:
            left_t_gini = gini(D.loc[left.index], flag)
            right_t_gini = gini(D.loc[right.index], flag)
            t_gini_index = (len(left) / D.shape[0]) * left_t_gini + (len(right) / D.shape[0]) * right_t_gini
            if t_gini_index < min_gini_index:
                min_gini_index = t_gini_index
                best_t = t
    return min_gini_index, best_t


class CART(object):
    def __init__(self, D, max_depth=5, prepruning=True) -> None:
        self.max_depth = max_depth
        self.prepruning = prepruning
        self.y = D['好瓜']
        self.y_pre = pd.Series(['好瓜'] * D.shape[0])
        super().__init__()

    def cal_weights(self, D):
        good_df = D.loc[D['好瓜']=='好瓜']
        bad_df = D.loc[D['好瓜']=='坏瓜']

        if good_df.shape[0] == 0:
            good_weights_sum = 0
        else:
            good_weights_sum = sum(good_df['权重'])

        if bad_df.shape[0] == 0:
            bad_weights_sum = 0
        else:
            bad_weights_sum = sum(bad_df['权重'])

        if good_weights_sum >= bad_weights_sum:
            return '好瓜'
        else:
            return '坏瓜'

    def generate_tree(self, D, p_node: Node):
        # 判断深度
        if p_node.depth >= self.max_depth:
            p_node.v = self.cal_weights(D)
            p_node.predict = self.cal_weights(D)
            return p_node

        # 判断 样本集合是否都是同一样本
        D_unique = D.drop_duplicates(keep=False)
        if D_unique.shape[0] == 0:
            # 样本全部相同，返回树
            p_node.v = D['好瓜'].value_counts().index[0]  # 返回标签频次最高对应的值
            p_node.predict = D['好瓜'].value_counts().index[0]
            return p_node

        # 判断 样本中标签是否完全相同
        if D['好瓜'].nunique() == 1:
            # 所有样本对应标签都一致，返回树
            p_node.v = D['好瓜'].values[0]
            p_node.predict = D['好瓜'].values[0]
            return p_node

        min_gini_index = float(inf)
        best_attr = 'x'
        best_t = 'y'

        # 获取没有划分过的属性
        X_columns = []
        for attr in self.attr_value_dict.keys():
            if self.attr_value_dict[attr]['used'] == False:
                X_columns.append(attr)

        # 划分选取
        for _, a in enumerate(X_columns):  # 逐个属性进行计算gini指数
            a_gini_value, t = gini_index(D, a)
            if a_gini_value < min_gini_index:
                min_gini_index = a_gini_value
                best_attr = a
                best_t = t

        if best_attr == 'x':
            raise f"划分选取异常"
        else:
            if best_t == '离散':
                p_node.v = f"{best_attr}=?"

                # 找出在属性best_attr上，取值为空的样本
                Dv_null_df = D.loc[D[best_attr] == 'NULL']
                for a in self.attr_value_dict[best_attr]['value']:
                    Dv = D.loc[D[best_attr] == a, :]
                    # 求出rv，在a上，取值为av，且在a上无缺失的样本比例
                    av_size = Dv.shape[0]
                    D_null_size = D.loc[D[best_attr] != 'NULL', :].shape[0]
                    rv = av_size / D_null_size

                    if Dv.shape[0] == 0:
                        # 最优划分上，无对应取值，该分支达到叶子节点
                        next_node = Node(f"{best_attr} = {a}", parent=p_node, av=a, aname=best_attr)
                        next_node.v = self.cal_weights(D)
                        next_node.predict = self.cal_weights(D)
                    else:
                        # 继续划分
                        Dv_null_df.loc[:, '权重'] = Dv_null_df['权重'] * rv
                        Dv_next = pd.concat([Dv, Dv_null_df])

                        next_node = Node(f"{best_attr} = {a}", parent=p_node, av=a, aname=best_attr)
                        next_node.predict = self.cal_weights(Dv_next)
                        self.generate_tree(Dv_next, next_node)

            if isinstance(best_t, float):
                # 连续属性，生成n-1个类后，调用generate_tree
                p_node.v = best_attr + "<=" + str(best_t) + "?"  # 节点信息
                print(best_attr, best_t, '连续属性')

                # 按最优划分点best_t将样本分割为两部分
                Dleft = D.loc[D[best_attr] <= best_t]
                Dright = D.loc[D[best_attr] > best_t]

                next_node_left = Node(p_node.v + '是', parent=p_node, av=best_t, aname=best_attr)
                next_node_right = Node(p_node.v + '否', parent=p_node, av=best_t, aname=best_attr)

                self.generate_tree(Dleft, next_node_left)  # 左边递归生成子树，是 yes 分支
                self.generate_tree(Dright, next_node_right)  # 同上。 注意，在此时没有将对应的A中值变成 -1
        return p_node

    def fit(self, D):
        """
        拟合CART
        """
        # 获取每个attr上，对应的取值
        attr_list = [x for x in list(D.columns) if not ((x == '好瓜') | (x == '权重'))]
        self.attr_value_dict = {}
        for attr in attr_list:
            # 此处标记，为体现算法的大致思路，若属性全部使用，则树生成完毕。、
            # 也可根据用depth来判断树是否生成结束
            not_null_value_list = D[attr][D[attr]!='NULL'].unique()
            self.attr_value_dict[attr] = {'value': not_null_value_list, 'used': False}

        root_node = Node('root')
        my_tree = self.generate_tree(D, root_node)
        return my_tree


def clf_data(node, d):
    target = 'x'
    if node.is_root:
        # 若node为根结点
        for child in node.children:
            if d[child.aname] == child.av:
                # 若满足child的取值条件，则继续向下
                target = clf_data(child, d)
                if target != 'x':
                    break  # 若已经找到结点，则退出循环，进递归
    else:
        # 若不是root结点
        if d[node.aname] == node.av:
            # 若与node的取值相同，则继续往下
            if node.is_leaf:
                target = node.predict
            else:
                for child in node.children:
                    target = clf_data(child, d)
                    if target != 'x':
                        break
    return target


def tree_accuracy(my_tree, D):
    y_predict = []
    for _, row in D.iterrows():
        _p = clf_data(my_tree, row)
        y_predict.append(_p)
    y_predict = pd.Series(y_predict)
    return len(y_predict[y_predict==D['好瓜']]) / D.shape[0]


def find_deepest_nodes(my_tree):
    leaves = findall(my_tree, lambda node: (node.is_leaf is True))
    max_depth = -1
    for leaf in leaves:
        if leaf.depth > max_depth:
            max_depth = leaf.depth
    return findall(my_tree, lambda node: (node.depth == max_depth))


def post_prune(my_tree, D):
    # 1. 找出最深叶结点
    while True:
        deepest_leaves = find_deepest_nodes(my_tree)
        if_imporve = False
        for leaf in deepest_leaves:
            if leaf.is_root:
                # 说明该叶子结点的父节点已经被判断过，且已经删除
                continue
            else:
                # 尝试删除其父结点所有分支，并计算准确率。若准确率提升则执行，若准确率下降则不执行
                org_accuracy = tree_accuracy(my_tree, D)
                leaf_parent = leaf.parent
                buffer_children = leaf.parent.children
                leaf.parent.children = []
                new_accuracy = tree_accuracy(my_tree, D)
                if new_accuracy < org_accuracy:
                    # 若令父节点为叶子结点后，准确率下降，则恢复父节点的children
                    leaf_parent.children = buffer_children
                else:
                    if_imporve = True
        if if_imporve is False:
            # 若无提升，则退出循环，返回树
            break
    return my_tree


df = dataset.load_watermelon_2()
root_node = Node('root')
my_tree = CART(df).fit(df)

# 以下为了测试post prune，强制替换结点的v成错误的结果
# 若如下操作后，原本模型的准确率为1，现在为0.88
# 且若删除该结点的父节点，并令其父节点作为叶子结点后，准确率上升为0.94
node_1 = findall(my_tree, lambda node: node.name == '触感 = 硬滑')[0]
node_2 = findall(my_tree, lambda node: node.name == '触感 = 软粘')[0]
node_1.v, node_1.predict = '坏瓜', '坏瓜'
node_2.v, node_2.predict = '好瓜', '好瓜'
print(RenderTree(my_tree))
print(tree_accuracy(my_tree, df))
print("======================> post pruned tree.")
new_my_tree = post_prune(my_tree, df)
print(RenderTree(new_my_tree))
print(tree_accuracy(new_my_tree, df))
