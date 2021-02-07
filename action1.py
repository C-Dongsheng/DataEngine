#!/usr/bin/env python3.6
# -*- coding: UTF-8 -*-
"""
Author: CaiDongsheng
Date: 2021/2/5 11:25
software: PyCharm

"""

import requests
from bs4 import BeautifulSoup
import pandas as pd

def get_page_content(url):
    # 得到页面的内容
    headers={'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36'}
    html=requests.get(url,headers=headers,timeout=10)
    content = html.text
    # print(content)
    # 通过content创建BeautifulSoup对象
    soup = BeautifulSoup(content, 'html.parser', from_encoding='utf-8')
    return soup

def content_analysis(soup):
    # 找到完整的投诉信息框
    temp = soup.find('div', class_="tslb_b")
    # 创建DataFrame
    df = pd.DataFrame(columns=['id', 'brand', 'car_model', 'type', 'desc', 'problem', 'datetime', 'status'])
    # 找出所有行记录（tr）
    tr_list = temp.find_all('tr')
    for tr in tr_list:
        # 取汽车投诉信息
        td_list = tr.find_all('td')
        # 如果没有td，就是表头th
        if len(td_list) > 0:
            # 投诉编号	投诉品牌	投诉车系	投诉车型	问题简述	典型问题	投诉时间	投诉状态
            id, brand, car_model, type, desc, problem, datetime, status = td_list[0].text, td_list[1].text, td_list[2].text, \
                                                                          td_list[3].text, td_list[4].text, td_list[5].text, \
                                                                          td_list[6].text, td_list[7].text
            # 数据放入字典
            temp = {}
            temp["id"] = id
            temp["brand"] = brand
            temp["car_model"] = car_model
            temp["type"] = type
            temp["desc"] = desc
            temp["problem"] = problem
            temp["datetime"] = datetime
            temp["status"] = status
            df = df.append(temp, ignore_index=True)
    return df

if __name__ == '__main__':
    # 创建DataFrame
    result = pd.DataFrame(columns=['id', 'brand', 'car_model', 'type', 'desc', 'problem', 'datetime', 'status'])
    # 请求URL
    base_url = 'http://www.12365auto.com/zlts/0-0-0-0-0-0_0-0-0-0-0-0-0-'
    page_num = 10
    for i in range(page_num):
        url = base_url + str(i + 1) + ".shtml"
        soup = get_page_content(url)
        df = content_analysis(soup)
        result = result.append(df)
    print(result)
    result.to_excel("test.xlsx")
