#!/usr/bin/env python3.6
# -*- coding: UTF-8 -*-
"""
Author: CaiDongsheng
Date: 2021/3/7 15:24
software: PyCharm

"""

import pandas as pd
from efficient_apriori import apriori

def read_data(file):
    """
    数据探索
    :return: 原始数据
    """
    # 读取数据
    data = pd.read_csv(file, header=None)
    # 数据探索
    print(data.shape)   #查看数据形状
    print(data.head())   #查看前5行数据
    return data

def get_transactions(dataset):
    """
    将数据存放到transactions中
    :param dataset: 原始数据
    :return: transactions
    """
    transactions = []
    for i in range(0, dataset.shape[0]):
        temp = []
        for j in range(0, 20):
            if str(dataset.values[i, j]) != 'nan':
                temp.append(str(dataset.values[i, j]))
        transactions.append(temp)
    return transactions

if __name__ == '__main__':
    # 加载数据
    data = read_data("Market_Basket_Optimisation.csv")
    # 获取transactions
    transactions = get_transactions(data)
    print(transactions)
    # 挖掘频繁项集和频繁规则
    itemsets, rules = apriori(transactions, min_support=0.02,  min_confidence=0.4)
    print("频繁项集：", itemsets)
    print("关联规则：", rules)