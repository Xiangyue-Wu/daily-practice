import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('movies.csv')

print(f'重复行数量: {df.duplicated().sum()}')

df.drop_duplicates(inplace=True)
print(f'剩余重复行: {df.duplicated().sum()}\n')

df['STARS']

# 清除冗余字符，分离演员与导演
datas = df['STARS'].str.replace('\n', '', regex=True).str.strip().str.replace(r'\s*,\s*', ', ', regex=True).str.split('|', expand=True)
mask = datas[1].isnull()
tmp = datas[0][mask]
datas[1][mask] = tmp
df['STARS'] = datas[1].str.replace('Stars:', '', regex=True).str.replace('Star:', '', regex=True).str.strip()
# 插入导演列
df.insert(6, 'DIRECTOR', datas[0].str.replace('Directors:', '', regex=True).str.replace('Director:', '', regex=True).str.strip())
df['DIRECTOR'][mask] = 'Unkown'
# 替换空字符串为Unkown
df['STARS'] = df['STARS'].str.replace(r'^$', 'Unkown', regex=True)

df

df['YEAR']

df['YEAR'] = df['YEAR'].astype(str)
# 选取初始年份
df['YEAR'] = df['YEAR'].str.extract(r'(\d{4})')
# 设定未知年份为-1
df['YEAR'] = df['YEAR'].fillna(-1)

df

df['GENRE']

df['GENRE'] = df['GENRE'].astype(str).str.replace('\n', '', regex=True).str.strip().str.replace(r'\s*,\s*', ', ', regex=True)
# 原始数据中未知类别为空行以及nan，进行统一
df['GENRE'] = df['GENRE'].fillna('Unkonwn')
df['GENRE'][df['GENRE'] == 'nan'] = 'Unknown'

df['GENRE']

df['RATING']

# 根据导演以及类别进行未知评分估计
df['RATING'] = df.groupby(['DIRECTOR', 'GENRE'])['RATING'].transform(lambda x: x.fillna(x.median(skipna=True)))
overall_median = df['RATING'].median(skipna=True)
df['RATING'] = df['RATING'].fillna(overall_median)

print(f'剩余未知评分: {df['RATING'].isnull().sum()}')

df['ONE-LINE']

df['ONE-LINE'] = df['ONE-LINE'].astype(str).str.replace('\n', '', regex=True).str.strip().str.replace(r'\s*,\s*', ', ', regex=True)
df['ONE-LINE'] = df['ONE-LINE'].replace('Add a Plot', 'Unkown')

df['ONE-LINE']

df['VOTES']

# 将原始的字符串形式的数据，去除逗号后，转换为数字
df['VOTES'] = df['VOTES'].astype(str).str.replace(',', '').str.strip()
df['VOTES'] = pd.to_numeric(df['VOTES'], errors='coerce')
df['VOTES'] = df['VOTES'].fillna(0)

df['VOTES']

df['Gross']

# 票房数据中null的数据较多，不便于观察，查看不为null的数据
df['Gross'][df['Gross'].notnull()]

df['Gross']=df['Gross'].str.replace('$', '').str.strip()
# 将M、K、B等单位转换为数量级
df['Gross']=df['Gross'].apply(lambda x: np.nan if pd.isna(x) else float(str(x)[:-1]) * (1e6 if 'M'in x else 1e3 if 'K' in x else 1e9 if 'B' in x else 1))

df['Gross'][df['Gross'].notnull()]

df['Gross'] = df['Gross'].fillna(0)

df['RunTime']

# 根据类别进行未知时长估计
df['RunTime'] = df.groupby(df['GENRE'].replace(',','',regex=True).str.strip())['RunTime'].transform(lambda x: x.fillna(x.median()))
df['RunTime']=df['RunTime'].fillna(df['RunTime'].median())

df['RunTime']

df.to_csv('movies_output.csv', index=False)
df = pd.read_csv('movies_output.csv')

df.info()

top10 = df.sort_values(by='RATING', ascending=False).head(10)

top10

import seaborn as sns
import matplotlib.pyplot as plt

# 设定字体，以显示中文
plt.rcParams['font.family'] = ['Times', 'SimHei']
plt.figure(figsize=(10, 6))
sns.barplot(data=top10, y='MOVIES', x='RATING', edgecolor='black', facecolor='#8EBAD9', errorbar='sd')
# 题外话，Dexter确实好看
plt.title('评分最高的10个电影')
plt.xlabel('评分')
plt.ylabel('电影')
plt.xlim(8, 10)
plt.show()

sns.histplot(df['RATING'], bins=20, kde=True)
plt.title('评分直方图')
plt.xlabel('评分')
plt.ylabel('数量')
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7, axis='y')
plt.show()

from collections import Counter

all_genres = df['GENRE'].str.split(', ')
genres = []
for sublist in all_genres:
    if isinstance(sublist, list):
        for genre in sublist:
            genres.append(genre)

genre_count = Counter(genres)
genre_count

plt.figure(figsize=(10,6))
df_vis = pd.DataFrame(genre_count.items(), columns=['Genre', 'Count'])
ax = sns.barplot(data=df_vis.head(10), y='Genre', x='Count', edgecolor='black', facecolor='#8EBAD9')
ax.bar_label(ax.containers[0], fontsize=10, padding=3)
plt.xlabel('电影数量')
plt.ylabel('类别')
plt.title('最常见的类别')
plt.show()

df_explode = df.dropna(subset=['GENRE']).copy()
df_explode['GENRE'] = df_explode['GENRE'].str.split(', ')
df_explode = df_explode.explode('GENRE')
df_explode = df_explode[df_explode['GENRE'] != 'Unknown']

plt.figure(figsize=(12, 6))
sns.catplot(data=df_explode, x='RunTime', y='GENRE', kind='violin', inner='point', palette='pastel', linewidth=0.2, bw_adjust=0.5)
plt.xlabel('时长')
plt.ylabel('类别')
plt.xlim(-10, 160)
plt.title('各种类别电影的时长')
plt.show()

df['YEAR'][df['YEAR'] != -1].value_counts().sort_index().plot(kind='line')
plt.xlabel('年份')
plt.ylabel('电影数量')
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7, axis='y')
plt.ylim(-50, 1800)
plt.show()
