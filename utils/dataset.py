# -*- encoding: utf-8 -*-
'''
@File    :   dataset.py
@Time    :   2021/07/06 15:16:32
@Author  :   qiujiayu 
@Version :   1.0
@Contact :   qiujy@highlander.com.cn
@Desc    :   创建数据集
'''

# here put the import lib
import pandas as pd
import numpy as np


def load_watermelon() -> pd.DataFrame:
    """生成西瓜数据

    Returns:
        pd.DataFrame: [西瓜数据]
    """
    # 色泽
    color_list = ['青绿', '乌黑', '乌黑', '青绿', '浅白', '青绿', '乌黑', '乌黑', 
                  '乌黑', '青绿', '浅白', '浅白', '青绿', '浅白', '乌黑', '浅白',
                  '青绿']
    
    # 根蒂
    root_list = ['蜷缩', '蜷缩', '蜷缩', '蜷缩', '蜷缩', '稍蜷', '稍蜷', '稍蜷',
                 '稍蜷', '硬挺', '硬挺', '蜷缩', '稍蜷', '稍蜷', '稍蜷', '蜷缩',
                 '蜷缩']
    
    # 敲声
    sound_list = ['浊响', '沉闷', '浊响', '沉闷', '浊响', '浊响', '浊响', '浊响',
                  '沉闷', '清脆', '清脆', '浊响', '浊响', '沉闷', '浊响', '浊响',
                  '沉闷']

    # 纹理
    texture_list = ['清晰', '清晰', '清晰', '清晰', '清晰', '清晰', '稍糊', '清晰',
                    '稍糊', '清晰', '模糊', '模糊', '稍糊', '稍糊', '清晰', '模糊',
                    '稍糊']
    
    # 脐部
    navel_list = ['凹陷', '凹陷', '凹陷', '凹陷', '凹陷', '稍凹', '稍凹', '稍凹',
                  '稍凹', '平坦', '平坦', '平坦', '凹陷', '凹陷', '稍凹', '平坦',
                  '稍凹']

    # 触感
    touch_list = ['硬滑', '硬滑', '硬滑', '硬滑', '硬滑', '软粘', '软粘', '硬滑',
                  '硬滑', '软粘', '硬滑', '软粘', '硬滑', '硬滑', '软粘', '硬滑',
                  '硬滑']

    # 是否好瓜，1: 是，0: 否
    target_list = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # init df
    df = pd.DataFrame(columns=['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', 'target'])
    df['色泽'] = color_list
    df['根蒂'] = root_list
    df['敲声'] = sound_list
    df['纹理'] = texture_list
    df['脐部'] = navel_list
    df['触感'] = touch_list
    df['target'] = target_list
    return df
