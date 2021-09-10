# -*- encoding: utf-8 -*-
'''
@File    :   203_CART.py
@Time    :   2021/08/02 17:17:10
@Author  :   qiujiayu 
@Version :   1.0
@Contact :   qiujy@highlander.com.cn
@Desc    :   CART demo 考虑了缺失值的情况与预剪枝
             但是，利用书中的缺失值数据的情况，那么在敲声样本会出现全部好瓜现象。分析后，是数据问题导致，利用无缺失情况，可正确分类。

             TODO: 添加后剪枝
'''

# here put the import lib
from math import inf
import pandas as pd
import numpy as np

from pprint import pprint
from collections import Counter
import utils.dataset as dataset
import matplotlib.pyplot as plt
import matplotlib as mpl


class Node(object):
    def __init__(self, title):
        self.title = title     # 上一级指向该节点的线上的标记文字
        self.v = 1              # 节点的信息标记
        self.children = []     # 节点的孩子列表
        self.deep = 0         # 节点深度
        self.ID = -1         # 节点编号
        self.leaf = False
        self.y_pre = []


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
        y = D['好瓜']
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
    def __init__(self, D, depth=5, prepruning=True) -> None:
        self.depth = depth
        self.prepruning = prepruning
        self.y = D['好瓜']
        self.y_pre = pd.Series(['好瓜'] * D.shape[0])
        super().__init__()

    def update_y_pre(self, idx, t):
        """
        更新y预测值

        Args:
            idx:
            t:

        Returns:

        """
        self.y_pre[idx] = t

    def cal_acc(self, y_pre=None):
        """
        计算模型准确率
        Returns:

        """
        if y_pre is None:
            b = self.y == self.y_pre
            return len(b[b==True]) / len(self.y)
        else:
            b = self.y == y_pre
            return len(b[b==True]) / len(self.y)

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
    
    def generate_tree(self, D, p_acc: float, p_value: str, p_node: Node):
        """生成树

        Args:
            D ([type]): [样本集合]
            p_acc ([float]): [父节点模型的准确率]
            p_value ([str]): [父节点取值]

        Returns:
            [type]: [description]
        """
        # init tree
        node = Node(p_value,)
        node.y_pre = pd.Series(['好瓜'] * len(self.y))
        node.deep = p_node.deep + 1

        # 判断深度
        if node.deep >= self.depth:
            node.v = self.cal_weights(D)
            node.leaf = True
            node.y_pre = self.y_pre.copy()
            return node

        # 判断 样本集合是否都是同一样本
        D_unique = D.drop_duplicates(keep=False)
        if D_unique.shape[0] == 0:
            # 样本全部相同，返回树
            node.v = D['好瓜'].value_counts().index[0]  # 返回标签频次最高对应的值
            node.leaf = True
            node.y_pre = self.y_pre.copy()
            return node
        
        # 判断 样本中标签是否完全相同
        if D['好瓜'].nunique() == 1:
            # 所有样本对应标签都一致，返回树
            node.v = D['好瓜'].values[0]
            node.leaf = True
            node.y_pre = self.y_pre.copy()
            return node

        # 计算样本中gini指数最小的属性，作为最优划分属性
        min_gini_index = float(inf)
        best_attr = 'x'

        # 获取没有划分过的属性
        X_columns = []
        for attr in self.attr_value_dict.keys():
            if self.attr_value_dict[attr]['used'] == False:
                X_columns.append(attr)

        for _, a in enumerate(X_columns):  # 逐个属性进行计算gini指数
            X = D.loc[:, X_columns]
            y = D['好瓜']
            a_gini_value, t = gini_index(D, a)
            if a_gini_value < min_gini_index:
                min_gini_index = a_gini_value
                best_attr = a
                best_t = t
        
        if best_attr != 'x':
            # 判断是否是离散属性
            if best_t == '离散':
                # 离散属性
                # 成功获取最优划分
                node.v = f"{best_attr}=?"

                # 对划分中的每个取值，再生成节点
                # 找出在属性best_attr上，取值为空的样本
                Dv_null_df = D.loc[D[best_attr]=='NULL']
                for a in self.attr_value_dict[best_attr]['value']:
                    Dv = D.loc[D[best_attr]==a, :]
                    # 求出rv，在a上，取值为av，且在a上无缺失的样本比例
                    av_size = Dv.shape[0]
                    D_null_size = D.loc[D[best_attr]!='NULL', :].shape[0]
                    rv = av_size / D_null_size

                    if Dv.shape[0] == 0:
                        # 最优划分上，无对应取值，该分支达到叶子节点
                        next_node = Node(a)
                        next_node.v = self.cal_weights(D)
                        self.update_y_pre(list(D.index), next_node.v)
                        next_node.leaf = True
                        next_node.y_pre = self.y_pre.copy()
                        node.children.append(next_node)
                    else:
                        # 继续划分
                        # self.attr_value_dict[best_attr]['used'] = True

                        # 检查该属性使用后，是否再无可用属性
                        X_columns = []
                        for attr in self.attr_value_dict.keys():
                            if self.attr_value_dict[attr]['used'] == False:
                                X_columns.append(attr)

                        if len(X_columns) == 0:
                            next_node = Node(a)
                            next_node.v = self.cal_weights(Dv)
                            self.update_y_pre(Dv.index, next_node.v)
                            next_node.leaf = True
                            next_node.y_pre = self.y_pre.copy()
                            node.children.append(next_node)
                        else:
                            # 更新存在缺失值的样本权重
                            Dv_null_df.loc[:, '权重'] = Dv_null_df['权重'] * rv
                            Dv_next = pd.concat([Dv, Dv_null_df])
                            t = self.cal_weights(Dv)

                            # 若按该属性取值划分后，模型的整理准确率（预剪枝）
                            if self.prepruning == True:
                                self.update_y_pre(list(Dv.index), t)
                                acc = self.cal_acc()
                            else:
                                acc = 1

                            next_node = Node(a)
                            if acc < p_acc:
                                next_node.v = t
                                next_node.leaf = True
                                next_node.y_pre = self.y_pre.copy()
                                node.children.append(next_node)
                            else:
                                next_node.deep = node.deep
                                node.children.append(self.generate_tree(Dv_next, acc, a, next_node))
            else:
                # 连续属性，生成n-1个类后，调用generate_tree
                node.v = best_attr + "<=" + str(best_t) + "?"  # 节点信息
                print(best_attr, best_t, '连续属性')

                # 按最优划分点best_t将样本分割为两部分
                Dleft = D.loc[D[best_attr]<=best_t]
                Dright = D.loc[D[best_attr]>best_t]

                # 计算left分割后，对应的样本准确率
                left_t = self.cal_weights(Dleft)
                right_t = self.cal_weights(Dright)

                next_node_left = Node('是')
                next_node_right = Node('否')
                if self.prepruning == True:
                    self.update_y_pre(list(Dleft.index), left_t)
                    self.update_y_pre(list(Dright.index), right_t)
                    acc = self.cal_acc()
                else:
                    acc = 1
                node.y_pre = self.y_pre.copy()
                next_node_left.y_pre = self.y_pre.copy()
                next_node_right.y_pre = self.y_pre.copy()
                node.children.append(self.generate_tree(Dleft, acc, "是", next_node_left))  # 左边递归生成子树，是 yes 分支
                node.children.append(self.generate_tree(Dright, acc, "否", next_node_right))  # 同上。 注意，在此时没有将对应的A中值变成 -1
        return node

    def postpruning(self, parent_node):
        # TODO: 完成postpruning.
        print(parent_node.v, parent_node.title, parent_node.y_pre)
        print(" =========================> ")
        parent_node_acc = self.cal_acc(parent_node.y_pre)
        for child in parent_node.children:
            if len(child.children) > 0:
                self.postpruning(child)
            else:
                child_acc = self.cal_acc(child.y_pre)
                if child_acc >= parent_node_acc:
                    # print(child.v, child.title, child.y_pre, child.leaf)
                    pass
                else:
                    print(child.v, child.title, child.y_pre, child.leaf)
                    pass
    
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

        root_acc = self.cal_acc()
        root_node = Node('root')
        my_tree = self.generate_tree(D, root_acc, 'root', root_node)
        self.postpruning(my_tree)
        return my_tree


def countLeaf(root,deep):
    root.deep = deep
    res = 0
    if root.v=='好瓜' or root.v=='坏瓜':   # 说明此时已经是叶子节点了，所以直接返回
        res += 1
        return res,deep
    curdeep = deep             # 记录当前深度
    for i in root.children:    # 得到子树中的深度和叶子节点的个数
        a,b = countLeaf(i,deep+1)
        res += a
        if b>curdeep: curdeep = b
    return res,curdeep
 
def giveLeafID(root,ID):         # 给叶子节点编号
    if root.v=='好瓜' or root.v=='坏瓜':
        root.ID = ID
        ID += 1
        return ID
    for i in root.children:
        ID = giveLeafID(i,ID)
    return ID
 
def plotNode(nodeTxt,centerPt,parentPt,nodeType):     # 绘制节点
    plt.annotate(nodeTxt,xy = parentPt,xycoords='axes fraction',xytext=centerPt,
                 textcoords='axes fraction',va="center",ha="center",bbox=nodeType,
                 arrowprops=arrow_args)
 
def dfsPlot(root):
    if root.ID==-1:          # 说明根节点不是叶子节点
        childrenPx = []
        meanPx = 0
        for i in root.children:
            cur = dfsPlot(i)
            meanPx += cur
            childrenPx.append(cur)
        meanPx = meanPx/len(root.children)
        c = 0
        for i in root.children:
            nodetype = leafNode
            if i.ID<0: nodetype=decisionNode
            plotNode(i.v,(childrenPx[c],0.9-i.deep*0.8/deep),(meanPx,0.9-root.deep*0.8/deep),nodetype)
            plt.text((childrenPx[c]+meanPx)/2,(0.9-i.deep*0.8/deep+0.9-root.deep*0.8/deep)/2,i.title)
            c += 1
        return meanPx
    else:
        return 0.1+root.ID*0.8/(cnt-1)


df = dataset.load_watermelon_3()
df.loc[:, '权重'] = [1] * df.shape[0]
df.fillna('NULL', inplace=True)

cart = CART(df, depth=5)
my_tree = cart.fit(df)
cnt, deep = countLeaf(my_tree,0)     # 得到树的深度和叶子节点的个数
print(cnt, deep)

# 绘制决策树
plt.rcParams['font.sans-serif'] = ['Simhei']
plt.rcParams['axes.unicode_minus'] = False

giveLeafID(my_tree,0)
decisionNode = dict(boxstyle = "sawtooth",fc = "0.9",color='blue')
leafNode = dict(boxstyle = "round4",fc="0.9",color='red')
arrow_args = dict(arrowstyle = "<-",color='green')
fig = plt.figure(1,facecolor='white')
rootX = dfsPlot(my_tree)
plotNode(my_tree.v,(rootX,0.9),(rootX,0.9),decisionNode)
plt.show()

