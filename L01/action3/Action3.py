#!/usr/bin/env python3.6
# -*- coding: UTF-8 -*-
"""
Author: CaiDongsheng
Date: 2021/1/28 16:55
software: PyCharm

"""


import pandas as pd

# 显示所有列
# pd.set_option("display.max_columns",None)
# 导入数据
result = pd.read_csv("car_complain.csv")
# print(result)

# 数据预处理
# 拆分problem类型=>多个字段。0->无关，1->有关
result = result.drop("problem", axis = 1).join(result.problem.str.get_dummies(','))
# print(result)

# 数据清洗，别名合并
def f(x):
    x = x.replace("一汽-大众","一汽大众")
    return x

# 品牌投诉总数
result['brand'] = result['brand'].apply(f)
df = result.groupby(['brand'])['id'].agg(['count']).sort_values('count', ascending = False)
df.reset_index(inplace=True)
print("==================品牌投诉总数，从大到小排序==================")
print(df)

# 车型投诉总数
df1 = result.groupby(['car_model'])['id'].agg(['count']).sort_values('count', ascending = False)
df1.reset_index(inplace=True)
print("==================车型投诉总数，从大到小排序==================")
print(df1)

# 品牌的平均车型投诉
df2 = result.groupby(['brand','car_model'])['id'].agg(['count'])
df2.reset_index(inplace=True)
# print(df2)
df2 = df2.groupby(['brand']).mean().sort_values('count', ascending = False)
df2.reset_index(inplace=True)
print("==================品牌的平均车型投诉，从大到小排序==================")
print(df2)
