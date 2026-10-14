import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

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

def split_data(df):
    """
    1. 删去演员和导演列均为空的数据
    2. 将没有评分的数据(pre)单独保存为pre.csv
    3. 将有评分的数据直接保存为data
    返回: pre, data 两个DataFrame
    """
    # 1. 删除 Director 与 Actors 同时为空的行
    df = df.dropna(subset=['Director', 'Actors'], how='all')

    # 2. 分离无评分数据
    pre = df[df['RATING'].isna()].copy()
    if not pre.empty:
        pre.to_csv(r'outputs/pre.csv', index=False, float_format='%.2f')
        print(f'无评分数据已保存为 pre.csv，共 {len(pre)} 行')
    else:
        print('未发现无评分数据')

    # 3. 分离有评分数据并直接保存为data
    data = df[df['RATING'].notna()].copy()
    if data.empty:
        print('未发现含评分数据')
        return pre, pd.DataFrame()

    # 保存含评分数据
    data.to_csv(r'outputs/data.csv', index=False, float_format='%.2f')
    print(f'含评分数据已保存为 data.csv，共 {len(data)} 行')

    return pre, data

def tfidf_directors_actors(df):
    """
    对 DataFrame 的“Director”和“Actors”列进行 TF-IDF 向量化。
    将每位导演或演员视为一个“词”，计算其在所有电影中的 TF-IDF 值。
    
    参数:
        df (pd.DataFrame): 包含“Director”和“Actors”列的 DataFrame。
    
    返回:
        scipy.sparse.csr_matrix: 合并后的 TF-IDF 稀疏矩阵（导演+演员）。
    """
    # 确保列存在
    if 'Director' not in df.columns or 'Actors' not in df.columns:
        raise ValueError("DataFrame 中缺少 'Director' 或 'Actors' 列")

    # 用空字符串填充 NaN，防止后续处理出错
    df['Director'] = df['Director'].fillna('')
    df['Actors']   = df['Actors'].fillna('')

    # 将导演和演员拼接成一段文本，用空格分隔
    # 注意：若导演或演员字段本身已用逗号分隔，可直接使用
    df['combined'] = (df['Director'].astype(str) + ' ' + df['Actors'].astype(str)).str.strip()

    # 初始化 TfidfVectorizer
    # 使用空格作为分隔符，将每个导演/演员视为一个“词”
    vectorizer = TfidfVectorizer(
        lowercase=False,      # 保留大小写（人名首字母大写）
        token_pattern=r'\b[^\s,]+\b',  # 按空格或逗号分隔
        max_features=None     # 不限制特征数量
    )

    # 拟合并转换
    tfidf_matrix = vectorizer.fit_transform(df['combined'])
    print(f'TF-IDF 向量化完成，特征数：{tfidf_matrix.shape[1]}')

    return tfidf_matrix

def split_and_vectorize(data, test_size=0.2, random_state=42):
    """
    1. 将 data（DataFrame）划分为训练集 train 和测试集 test
    2. 调用 tfidf_directors_actors 对 train 和 test 的 'Director' 与 'Actors' 列进行 TF-IDF 向量化
    3. 提取 y_train 和 y_test（即 RATING 列）
    
    返回:
        X_train_tfidf, X_test_tfidf, y_train, y_test
    """
    # 1. 划分训练集与测试集
    train, test = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state
    )
    
    # 2. 分别对训练集和测试集进行 TF-IDF 向量化
    X_train_tfidf = tfidf_directors_actors(train)
    X_test_tfidf  = tfidf_directors_actors(test)
    
    # 3. 提取标签
    y_train = train['RATING'].values
    y_test  = test['RATING'].values
    
    return X_train_tfidf, X_test_tfidf, y_train, y_test

def train_rf_with_cv(X_train_tfidf, y_train, cv=5, random_state=42):
    """
    1. 初始化随机森林回归器
    2. 使用随机搜索寻找最佳超参数
    3. 输出五折交叉验证最佳R²
    4. 返回交叉验证得分最高的模型
    """
    # 定义超参数搜索空间

    param_distributions = {
        'n_estimators': [200, 300, 400],       # 聚焦于中等数量的树，平衡性能与耗时
        'max_depth': [10, 20, 30, None],        # 包含None，允许树完全生长
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']       # 移除了不被支持的 'auto'，其等价于 'sqrt'
    }
    # 初始化随机森林
    rf = RandomForestRegressor(random_state=random_state)

    # 随机搜索 + 五折交叉验证
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=20,          # 随机尝试20组参数
        cv=cv,
        scoring='r2',
        random_state=random_state,
        n_jobs=-1
    )

    # 拟合
    random_search.fit(X_train_tfidf, y_train)

    # 输出最佳交叉验证R²
    best_score = random_search.best_score_
    print(f"随机搜索后五折交叉验证最佳平均 R²: {best_score:.4f}")

    # 输出所耗时间
    print(f"随机搜索耗时: {random_search.cv_results_['mean_fit_time'].mean():.2f} 秒")

    # 返回最佳模型
    return random_search.best_estimator_
    

# 在主流程中调用
if __name__ == "__main__":
    path = r'data/movies.csv'
    # 读取 
    df = rd_exp(path)
    # 将每一类的数据类型、格式统一
    df = clean_data(df)
    # 删除重复值
    df = del_re(df)
    # 对未知年份进行插值
    df = fill_missing_years(df)
    # 分离演员表中的导演与演员，以及对应评分
    df = separate_directors_actors(df)
    # 划分数据集
    pre, data = split_data(df)
    
    # 划分训练/测试集并完成 TF-IDF 向量化
    X_train_tfidf, X_test_tfidf, y_train, y_test = split_and_vectorize(data)
    
    # 训练随机森林并做五折交叉验证
    best_model = train_rf_with_cv(X_train_tfidf, y_train)
 