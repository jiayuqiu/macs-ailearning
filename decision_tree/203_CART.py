# -*- encoding: utf-8 -*-
'''
@File    :   203_CART.py
@Time    :   2021/08/02 17:17:10
@Author  :   qiujiayu 
@Version :   1.0
@Contact :   qiujy@highlander.com.cn
@Desc    :   CART demo 没有考虑缺失值与连续属性的情况
'''

# here put the import lib
from math import inf
import pandas as pd
import numpy as np

from pprint import pprint
from collections import Counter
from utils.dataset import load_watermelon_2, load_watermelon_2_alpha
import matplotlib.pyplot as plt
import matplotlib as mpl
# from sklearn.tree import DecisionTreeClassifier


class Node(object):
    def __init__(self, title):
        self.title = title     # 上一级指向该节点的线上的标记文字
        self.v = 'yyy'              # 节点的信息标记
        self.children = []     # 节点的孩子列表
        self.deep = 0         # 节点深度
        self.ID = -1         # 节点编号
        self.leaf = False


def get_rho(D, flag):
    """get rho by weights"""
    d_size = sum(D['权重'])
    a_not_null_size = sum(D.loc[D[flag]!='NULL']['权重'])
    return a_not_null_size / d_size


def get_pk(D, flag, target):
    """get pk"""
    a_not_null_size = sum(D.loc[D[flag]!='NULL']['权重'])
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
    return rho * gini_value


class CART(object):
    def __init__(self, depth=5) -> None:
        # self.shape = (X.shape[0], X.shape[1] + 1)
        # self.tree = list()
        self.depth = depth
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
    
    def generate_tree(self, D, attr_used_list: list, p_value: str):
        """生成树

        Args:
            D ([type]): [样本集合]
            attr_used_list ([list]): [已经被使用过的划分]
            p_value ([str]): [父节点取值]

        Returns:
            [type]: [description]
        """
        # init tree
        node = Node(p_value)

        # 判断 样本集合是否都是同一样本
        D_unique = D.drop_duplicates(keep=False)
        if D_unique.shape[0] == 0:
            # 样本全部相同，返回树
            node.v = D['好瓜'].value_counts().index[0]  # 返回标签频次最高对应的值
            print(node.v)
            node.leaf = True
            return node
        
        # 判断 样本中标签是否完全相同
        if D['好瓜'].nunique() == 1:
            # 所有样本对应标签都一致，返回树
            node.v = D['好瓜'].values[0]
            node.leaf = True
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
            a_gini_value = gini_index(D, a)
            if a_gini_value < min_gini_index:
                min_gini_index = a_gini_value
                best_attr = a
        
        if best_attr != 'x':
            # 成功获取最优划分
            node.v = f"{best_attr}=?"
            
            # 对划分中的每个取值，再生成节点
            # 找出在属性best_attr上，取值为空的样本
            Dv_null_df = D.loc[D[best_attr]=='NULL']
            for a in self.attr_value_dict[best_attr]['value']:
                print(a, type(a), a == '平坦')
                Dv = D.loc[D[best_attr]==a, :]
                # 求出rv，在a上，取值为av，且在a上无缺失的样本比例
                av_size = Dv.shape[0]
                D_null_size = D.loc[D[best_attr]!='NULL', :].shape[0]
                rv = av_size / D_null_size

                # 更新样本权重
                Dv_null_df.loc[:, '权重'] = Dv_null_df['权重'] * rv
                if Dv.shape[0] == 0:
                    # 最优划分上，无对应取值，该分支达到叶子节点
                    next_node = Node(a)
                    next_node.v = self.cal_weights(D)
                    print(next_node.v)
                    next_node.leaf = True
                    node.children.append(next_node)
                else:
                    # 继续划分
                    print(f"next parent name = {a}")
                    self.attr_value_dict[best_attr]['used'] = True

                    # 检查该属性使用后，是否再无可用属性
                    X_columns = []
                    for attr in self.attr_value_dict.keys():
                        if self.attr_value_dict[attr]['used'] == False:
                            X_columns.append(attr)

                    if len(X_columns) == 0:
                        next_node = Node(a)
                        next_node.v = self.cal_weights(D)
                        next_node.leaf = True
                        node.children.append(next_node)
                    else:
                        Dv_next = pd.concat([Dv, Dv_null_df])
                        node.children.append(self.generate_tree(Dv_next, [], a))
        return node
    
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
        print(self.attr_value_dict)
        
        attr_used_list = [1] * len(attr_list)
        my_tree = self.generate_tree(D, attr_used_list, 'root')
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
        print(f"root.v = {root.v}")
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


df = load_watermelon_2_alpha()
df.loc[:, '权重'] = [1] * df.shape[0]
df.fillna('NULL', inplace=True)
print(df)

cart = CART()
my_tree = cart.fit(df)
cnt, deep = countLeaf(my_tree,0)     # 得到树的深度和叶子节点的个数
print(cnt, deep)

def plot_tree(my_tree):
    for child in my_tree.children:
        print(f"parent = {my_tree.title}")
        if child.leaf:
            print(child.v)
        else:
            print(child.children)
            plot_tree(child)

# plot_tree(my_tree)
# for node in my_tree:
#     print(node)

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

