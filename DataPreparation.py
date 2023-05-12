## This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import pandas as pd
import re

df =  pd.read_csv("kaggle/results.csv")
df["date"] = pd.to_datetime(df["date"])
#统计空值的个数
#print(df.isna().sum())
#去掉空值
df.dropna(inplace=True)
#print(df.isna().sum())
#数据类型
print(df.dtypes)
#根据比赛时间倒排
#print(df.sort_values("date").tail())
#只读大于2018-08-01的比赛
df = df[(df["date"] >= "2018-8-1")].reset_index(drop=True)
print(df.sort_values("date").tail())
#主队-国家-比赛场数统计
print(df.home_team.value_counts())

#读取rank的比赛记录
rank = pd.read_csv("kaggle/fifa_ranking-2022-10-06.csv")
#只读大于2018-08-01的比赛
rank["rank_date"] = pd.to_datetime(rank["rank_date"])
rank = rank[(rank["rank_date"] >= "2018-8-1")].reset_index(drop=True)
#世界杯的一些球队在排名数据集中有不同的名字。因此，需要进行调整。
rank["country_full"] = rank["country_full"].str.replace("IR Iran", "Iran").str.replace("Korea Republic", "South Korea").str.replace("USA", "United States")

rank = rank.set_index(['rank_date']).groupby(['country_full'], group_keys=False).resample('D').first().fillna(method='ffill').reset_index()
