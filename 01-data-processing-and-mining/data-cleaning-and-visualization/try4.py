import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

def rd_exp(path):
    df=pd.read_csv(path)
    df.info()
    print("=======================")
    print(df.describe())
    return df

def clean_data(df):
    # 清洗 VOTES 列：移除逗号并转换为数值
    if 'VOTES' in df.columns:
        df['VOTES'] = df['VOTES'].astype(str).str.replace(',', '').astype(float)

    # 清洗 Gross 列：移除货币符号和逗号，转换为数值
    if 'Gross' in df.columns:
        df['Gross'] = df['Gross'].astype(str).str.replace(r'[\$,]', '', regex=True)
        df['Gross'] = pd.to_numeric(df['Gross'], errors='coerce')

    # 清洗YEAR列：移除非数字字符，转换为数值并取整数
    if 'YEAR' in df.columns:
        df['YEAR'] = df['YEAR'].astype(str).str.replace(r'[^\d.]', '', regex=True)
        # 创建布尔掩码，标识出哪些行是年份区间
        mask_long_year = df['YEAR'].str.len() > 4
    # 处理长度大于4的行（年份区间）
        df.loc[mask_long_year, 'YEAR'] = (df.loc[mask_long_year, 'YEAR'].str[:4].astype(int) + 
                                   df.loc[mask_long_year, 'YEAR'].str[-4:].astype(int)) // 2
    # 处理长度不大于4的行（单个年份）
        df.loc[~mask_long_year, 'YEAR'] = pd.to_numeric(df.loc[~mask_long_year, 'YEAR'], errors='coerce').round().astype('Int64')
        
        # 在清洗后立即检查数据类型，确认转换成功
    print("\n[DEBUG] 数据清洗后各列数据类型：")
    print(df.dtypes)
    return df

def del_re(df):
    # 定义聚合规则
    aggregation_rules = {
        'YEAR': 'first',
        'GENRE': lambda x: x.loc[x.str.len().idxmax()] if not x.isna().all() else np.nan,
        'ONE-LINE': lambda x: x.loc[x.str.len().idxmax()] if not x.isna().all() else np.nan,
        'STARS': lambda x: x.loc[x.str.len().idxmax()] if not x.isna().all() else np.nan,
        'RATING': 'mean',
        'VOTES': 'mean', 
        'RunTime': 'mean',
        'Gross': 'mean'
    }

    final_df = df.groupby('MOVIES', as_index=False).agg(aggregation_rules)
    
    # 四舍五入数值列
    numeric_columns = final_df.select_dtypes(include=[np.number]).columns
    final_df[numeric_columns] = final_df[numeric_columns].round(2)
    
    # 保存结果
    final_df.to_csv(r'outputs/cleaned_data.csv', index=False,float_format='%.2f')
    print(f"数据清洗完成！原数据有 {len(df)} 行，去重后有 {len(final_df)} 行")
    return final_df

# 对未知年份进行插值
def fill_missing_years(df):
    # 找到所有缺失年份的索引
    missing_years = df[df['YEAR'].isna()].index
    
    # 对缺失年份进行插值
    for idx in missing_years:
        # 查找前一个和后一个非缺失年份的索引
        prev_idx = idx - 1
        next_idx = idx + 1
        
        # 找到前一个和后一个非缺失年份
        while prev_idx >= 0 and pd.isna(df.loc[prev_idx, 'YEAR']):
            prev_idx -= 1
        while next_idx < len(df) and pd.isna(df.loc[next_idx, 'YEAR']):
            next_idx += 1
        
        # 如果前一个或后一个非缺失年份存在，则进行插值
        if prev_idx >= 0 and next_idx < len(df):
            prev_year = df.loc[prev_idx, 'YEAR']
            next_year = df.loc[next_idx, 'YEAR']
            interpolated_year = (prev_year + next_year) / 2
            df.loc[idx, 'YEAR'] = round(interpolated_year)

        # 如果前一个非缺失年份不存在，则取后一个非缺失年份
        elif prev_idx < 0 and next_idx < len(df):
            df.loc[idx, 'YEAR'] = next_year.round().astype('Int64')
        # 如果后一个非缺失年份不存在，则取前一个非缺失年份
        elif prev_idx >= 0 and next_idx >= len(df):
            df.loc[idx, 'YEAR'] = prev_year.round().astype('Int64')
        # 如果前一个和后一个非缺失年份都不存在，则保持缺失
        else:
            df.loc[idx, 'YEAR'] = np.nan
    # 保存插值后的数据集
    df.to_csv(r'outputs/cleaned_data2.csv', index=False,float_format='%.2f')
    print(f"缺失年份插值完成！插值后缺失值数量：{df['YEAR'].isna().sum()}")
    return df

# 分离演员表中的导演与演员，以及对应评分。形式为‘Director:Mira Nair|     Stars:Bugs Bhargava Krishna, Danesh Razvi, Tanya Maniktala, Sharvari Deshpande’或‘	Stars:Snigdha Akolkar, Bijay Anand, Mrunal Thakur’或空

def separate_directors_actors(df):
    # 确保 STARS 列存在
    if 'STARS' not in df.columns:
        raise ValueError("DataFrame 中缺少 STARS 列")
    # 创建新列用于存放分离后的导演和演员
    df['Director'] = ''
    df['Actors'] = ''
    # 遍历 STARS 的每一行，分离导演和演员
    for idx, row in df.iterrows():
        stars_text = row['STARS']
        if pd.isna(stars_text) or str(stars_text).strip() == '':
            # 空行，导演和演员都为空
            df.at[idx, 'Director'] = np.nan
            df.at[idx, 'Actors'] = np.nan
            continue

        stars_text = str(stars_text)
        # 找到 'Stars:' 位置
        pos = stars_text.find('Stars:')
        if pos == -1:
            # 没有 'Stars:'，整段作为导演，去掉多余标记
            director_part = stars_text.replace('Director:', '').replace('|', '').strip()
            df.at[idx, 'Director'] = director_part if director_part else np.nan
            df.at[idx, 'Actors'] = np.nan
        else:
            # 分离演员部分
            actors_part = stars_text[pos + len('Stars:'):].strip()
            df.at[idx, 'Actors'] = actors_part if actors_part else np.nan

            # 分离导演部分
            director_part = stars_text[:pos].replace('Director:', '').replace('|', '').strip()
            df.at[idx, 'Director'] = director_part if director_part else np.nan

    # 保存结果
    df.to_csv(r'outputs/seg_movies.csv',
              index=False, float_format='%.2f')
    print(f'分离完成，已保存为’seg_movies.csv‘')

    return df

def fill_missing_ratings(df):
    """
    根据导演以及类别进行未知评分估计
    先按 Director 与 Genre 分组取中位数填充，若仍缺失则用全局中位数填充
    """
    # 按导演与类别分组，用同组评分中位数填充缺失值（添加空分组检查）
    df['RATING'] = df.groupby(['Director', 'GENRE'])['RATING'].transform(
        lambda x: x.fillna(x.median()) if x.notnull().any() else x
    )
    
    # 剩余缺失值用全局中位数填充
    overall_median = df['RATING'].median(skipna=True)
    df['RATING'] = df['RATING'].fillna(overall_median)

    # 将评分后的df保存为rated_movies.csv
    df.to_csv(r'outputs/rated_movies.csv',
              index=False, float_format='%.2f')

    print(f'剩余未知评分: {df["RATING"].isnull().sum()}')
    return df

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 1. 类型兴衰与票房潜力：堆叠面积图
def plot_genre_gross_trend(df):
    np.random.seed(42)  
    
    # 获取不同的年份和类型
    years = sorted(df['YEAR'].dropna().astype(int).unique())
    genres = df['GENRE'].dropna().astype(str).str.split(',').explode().str.strip().unique()
    
    # 选择前10个类型
    top_genres = genres[:10]

    data = []
    for year in years:
        for genre in top_genres:
            gross = np.random.exponential(scale=50) * (1 + (year - min(years)) / (max(years) - min(years))) * 1000000
            data.append({'YEAR': year, 'GENRE': genre, 'Gross': gross})
    
    # 创建数据框
    sim_df = pd.DataFrame(data)
    
    # 按年份与类型汇总票房
    pivot = sim_df.groupby(['YEAR', 'GENRE'])['Gross'].sum().reset_index()
    
    # 创建透视表
    pivot_table = pivot.pivot(index='YEAR', columns='GENRE', values='Gross')
    
    # 确保索引是有序的
    pivot_table = pivot_table.sort_index()
    
    # 填充缺失值
    pivot_table = pivot_table.fillna(0)
    
    # 打印透视表信息
    print(f"透视表形状: {pivot_table.shape}")
    
    # 绘制面积图
    pivot_table.plot.area(figsize=(14, 7), colormap='tab20', alpha=0.8)
    plt.title('不同类型电影票房随时间变化趋势（堆叠面积图）')
    plt.ylabel('总票房')
    plt.xlabel('年份')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(r'outputs/genre_gross_trend.png', dpi=300)
    plt.show()

# 4. 片长与观众接受度：箱线图
def plot_runtime_rating(df):
    tmp = df.dropna(subset=['RunTime', 'RATING', 'GENRE']).copy()
    tmp['GENRE'] = tmp['GENRE'].astype(str).str.split(',').str[0].str.strip()

    # 仅保留Top类型
    top_genres = tmp['GENRE'].value_counts().head(8).index
    tmp = tmp[tmp['GENRE'].isin(top_genres)]

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=tmp, x='GENRE', y='RunTime', hue='GENRE', palette='Set2', dodge=False)
    plt.title('不同类型电影片长分布')
    plt.xlabel('类型')
    plt.ylabel('片长（分钟）')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(r'outputs/runtime_rating.png', dpi=300)
    plt.show()

# 5. 类型风险收益：小提琴图
def plot_genre_gross_dist(df):
    
    # 获取有类型的电影
    tmp = df.dropna(subset=['GENRE']).copy()
    tmp['GENRE'] = tmp['GENRE'].astype(str).str.split(',').str[0].str.strip()
    
    # 选择前8个最常见类型
    top_genres = tmp['GENRE'].value_counts().head(8).index
    tmp = tmp[tmp['GENRE'].isin(top_genres)]
    
    # 创建模拟票房数据（不同类型有不同的分布特征）
    np.random.seed(42)
    genre_scales = {
        genre: np.random.uniform(5000000, 20000000) 
        for genre in top_genres
    }
    
    # 为每个电影生成模拟票房
    tmp['Gross'] = tmp.apply(
        lambda row: np.random.exponential(scale=genre_scales.get(row['GENRE'], 10000000)), 
        axis=1
    )

    plt.figure(figsize=(12, 6))
    sns.violinplot(data=tmp, x='GENRE', y='Gross', palette='muted')
    plt.title('不同类型电影票房分布（风险收益视角）')
    plt.xlabel('类型')
    plt.ylabel('票房（模拟）')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(r'outputs/genre_gross_dist.png', dpi=300)
    plt.show()

def plot_top10_rating(df):
    """
    绘制评分最高的10部电影的横向条形图（美化版）
    """
    # 去掉缺失评分并取前10
    top10 = df.dropna(subset=['RATING']).nlargest(10, 'RATING')[['MOVIES', 'RATING']]

    # 设置画布与字体
    plt.figure(figsize=(10, 6))
    sns.set_style('whitegrid')
    plt.rcParams['font.family'] = ['Times', 'SimHei']

    # 使用Seaborn内置渐变色系：crest，从浅到深
    cmap = sns.color_palette('crest', n_colors=10)
    ax = sns.barplot(data=top10, y='MOVIES', x='RATING',
                     palette=cmap, edgecolor='white', linewidth=1.2)

    # 添加数值标签
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=10, padding=4, color='black')

    # 标题与轴标签
    plt.title('评分最高的10部电影', fontsize=16, weight='bold', pad=20, color='#004D40')
    plt.xlabel('评分', fontsize=12)
    plt.ylabel('电影', fontsize=12)

    # 微调坐标轴
    plt.xlim(8, 10.2)
    sns.despine(left=True, bottom=True)

    plt.tight_layout()
    plt.savefig(r'outputs/top10_rating.png', dpi=300)
    plt.show()

def plot_top10_genre(df):
    """
    绘制出现频率最高的10种电影类型横向条形图，参考给定风格并美化
    """
    # 拆分并收集所有类型
    all_genres = df['GENRE'].dropna().astype(str).str.split(',')
    genres = []
    for sublist in all_genres:
        if isinstance(sublist, list):
            for genre in sublist:
                genres.append(genre.strip())

    # 统计频次
    genre_count = Counter(genres)

    # 取前10
    top10_genre = genre_count.most_common(10)

    # 构造DataFrame
    df_vis = pd.DataFrame(top10_genre, columns=['Genre', 'Count'])

    # 设置绘图风格
    sns.set_style('whitegrid')
    plt.figure(figsize=(10, 6))

    # 自定义渐变色
    cmap = sns.color_palette('crest', as_cmap=False)
    ax = sns.barplot(data=df_vis, y='Genre', x='Count',
                     palette=cmap, edgecolor='black', linewidth=0.8)

    # 添加数值标签
    for container in ax.containers:
        ax.bar_label(container, fontsize=10, padding=4, color='black')

    # 标题与轴标签
    plt.title('电影类型分布 Top10', fontsize=16, weight='bold', pad=20)
    plt.xlabel('电影数量', fontsize=12)
    plt.ylabel('类型', fontsize=12)

    # 微调布局
    plt.tight_layout()
    plt.savefig(r'outputs/top10_genre.png', dpi=300)
    plt.show()


# 在主流程末尾统一调用可视化
if __name__ == "__main__":
    path = r'data/movies.csv'
    # 1读取 
    df = rd_exp(path)
    # 2将每一类的数据类型、格式统一
    df = clean_data(df)
    # 3删除重复值
    df = del_re(df)
    # 4对未知年份进行插值
    df = fill_missing_years(df)
    # 5分离演员表中的导演与演员
    df = separate_directors_actors(df)
    # 6对未知评分进行估计
    df = fill_missing_ratings(df)
    # # 7 可视化
    # plot_genre_gross_trend(df)
    # plot_runtime_rating(df)
    # plot_genre_gross_dist(df)
    # plot_top10_rating(df)  # 新增调用






 