#!/usr/bin/env python3.6
# -*- coding: UTF-8 -*-
"""
Author: CaiDongsheng
Date: 2021/2/28 15:24
software: PyCharm

"""

from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def import_data(file):
    """
    加载原始数据，探索数据结构
    :param file: 数据文件地址
    :return: DataFrame结构的原始数据
    """
    # 数据加载
    data = pd.read_csv(file, encoding='gbk')
    # 打印数据结构
    print(data.shape)
    print(data.head())
    return data

def data_cleaning(data):
    """
    数据清洗，产生训练集
    :param data: 原始数据
    :return: 训练集
    """

    # 处理训练集
    train_x = data[["人均GDP", "城镇人口比重", "交通工具消费价格指数", "百户拥有汽车量"]]
    # 规范化到 [0,1] 空间
    min_max_scaler = preprocessing.MinMaxScaler()
    train_x = min_max_scaler.fit_transform(train_x)
    # 对训练集数据进行降维处理,导入PCA
    from sklearn.decomposition import PCA
    # 设置主成分数量为2以便进行可视化
    pca = PCA(n_components=2)
    pca.fit(train_x)
    train_x = pca.transform(train_x)
    # 打印处理后的数据集形态
    print(train_x.shape)

    return train_x

def sse(train_data):
    """
    K-Means 手肘法：统计不同K取值的误差平方和
    :param train_data: 训练集
    :return:
    """
    sse = []
    for k in range(2, 10):
        # kmeans算法
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(train_data)
        # 计算inertia簇内误差平方和
        sse.append(kmeans.inertia_)
    x = range(2, 10)
    plt.xlabel('K')
    plt.ylabel('SSE')
    plt.plot(x, sse, 'o-')
    plt.show()

def silhouette(train_data):
    """
    K-Means 轮廓系数法：统计不同K取值的轮廓系数
    :param train_data: 训练集
    :return: 
    """
    from sklearn import metrics
    scores = []  #存放轮廓系数
    for k in range(2, 10):
        # kmeans算法
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(train_data)
        predict_y = kmeans.predict(train_data)   #预测聚类模型
        scores.append(metrics.silhouette_score(train_data,predict_y,metric='euclidean'))
    x = range(2, 10)
    plt.xlabel('K')
    plt.ylabel('scores')
    plt.plot(x, scores, 'o-')
    plt.show()

def K_means(train_data, k):
    """
    使用KMeans聚类,训练聚类模型
    :param train_data: 训练集
    :param k: 聚类数量
    :return: kmeans聚类模型
    """
    kmeans = KMeans(n_clusters=k, random_state=0)  # 建立模型对象
    kmeans.fit(train_data)  # 训练聚类模型
    # predict_y = kmeans.predict(train_data)  # 预测聚类模型
    return kmeans

def show_result(kmeans,k,result):
    """
    聚类可视化
    :param kmeans: kmeans聚类模型
    :param k: 聚类数量
    :param result: 降维表格数据与聚类结果合并数据
    :return:
    """

    centers = kmeans.cluster_centers_  # 类别中心
    print(centers)
    colors = ['r', 'c', 'b', 'g', 'y']
    plt.figure()
    for j in range(k):
        index = np.where(kmeans.labels_ == j)
        cluster = result.iloc[index]
        plt.scatter(cluster.iloc[:, 0], cluster.iloc[:, 1], c=colors[j], marker='.')
        plt.plot(centers[j][0], centers[j][1], 'o', markerfacecolor=colors[j], markeredgecolor='k',
                 markersize=8)  # 画类别中心
    plt.show()

if __name__ == '__main__':

    data = import_data("car_data.csv")
    train_x = data_cleaning(data)
    print(train_x)
    print(train_x.shape)
    # 通过手肘法和轮廓系数法，确定K值
    sse(train_x)
    silhouette(train_x)
    # 选定K=5，训练聚类模型，预测聚类模型，得到聚类结果
    kmeans = K_means(train_x,5)
    # 合并聚类结果，插入到原数据中
    result = pd.concat((data, pd.DataFrame(kmeans.labels_)), axis=1)
    result.rename({0: "聚类结果"}, axis=1, inplace=True)
    print(result)
    # 打印分组情况
    for item in sorted(result["聚类结果"].unique()):
        print('第{}组: '.format(item+1))
        # 筛选同组纪录
        record = result[result[ "聚类结果"]==item]
        print(record['地区'].values)

    # 聚类可视化
    # 合并聚类结果，插入到规范化数据中
    result = pd.concat((pd.DataFrame(train_x), pd.DataFrame(kmeans.labels_)), axis=1)
    show_result(kmeans,5,result)