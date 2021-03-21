#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2021/3/21 9:10
@Author  : Cai Dongsheng
@File    : action_prophet.py
@Software: PyCharm

"""
import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt

# 数据加载
df = pd.read_csv('./train.csv')
df = df[["Datetime","Count"]]

# 将时间作为df的索引
df.Datetime = pd.to_datetime(df.Datetime, format='%d-%m-%Y %H:%M')
df.index = df.Datetime
print(df.head())

# 数据探索
# 按照天来统计
df_day = df.resample('D').sum()
df_day = df_day.reset_index()
print(df_day)
# 修改列名 Datetime => ds, Count => y
df_day.rename(columns={'Datetime':'ds', 'Count':'y'}, inplace=True)
print(df_day)

# 拟合模型
model = Prophet(yearly_seasonality=True, seasonality_prior_scale=0.1)
model.fit(df_day)

# 构建待预测日期数据框，periods = 213 代表除历史数据的日期外再往后推 213 天
future = model.make_future_dataframe(periods=213)
#print(future.tail())

# 预测数据集
forecast = model.predict(future)
#print(forecast.columns)
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# 展示预测结果
model.plot(forecast)
plt.show()
