#!/usr/bin/env python3.6
# -*- coding: UTF-8 -*-
"""
Author: CaiDongsheng
Date: 2021/3/13 12:59
software: PyCharm

"""
import pandas as pd
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt


# 数据加载
data = pd.read_csv("Market_Basket_Optimisation.csv",header=None)
print(data)

# 将数据存放操transactions中
transactions = []
# 将数据存储到字典中
item_count = {}
for i in range(data.shape[0]):
	temp = []
	for j in range(data.shape[1]):
		item = str(data.values[i, j])
		# 提取非空数
		if item != "nan":
			temp.append(item)
			# 统计出现频次
			if item not in item_count:
				item_count[item] = 1
			else:
				item_count[item] += 1
	transactions.append(temp)
# print(transactions)
# print(len(transactions))

# 去掉停用词
def remove_stop_words(f):
	stop_words = []
	for stop_word in stop_words:
		f = f.replace(stop_word, '')
	return f

# 生成词云
def create_word_cloud(f):
	print('根据词频，开始生成词云!')
	# 去掉停用词
	f = remove_stop_words(f)
	# 分词
	cut_text = word_tokenize(f)
	#print(cut_text)
	cut_text = " ".join(cut_text)
	wc = WordCloud(
		max_words=100,
		width=2000,
		height=1200,
    )
	# 生成词语
	wordcloud = wc.generate(cut_text)
	# 写词云图片
	wordcloud.to_file("wordcloud.jpg")
	# 显示词云文件
	plt.imshow(wordcloud)
	plt.axis("off")
	plt.show()


# 生成词云
all_word = " ".join("%s" %item for item in transactions)
# print(all_word)
create_word_cloud(all_word)

# Top10的商品有哪些
print("Top10的商品有:")
print(sorted(item_count.items(),key=lambda x:x[1],reverse=True)[:10])