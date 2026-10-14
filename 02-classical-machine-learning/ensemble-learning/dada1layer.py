import pandas as pd
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder

def read_data(file_path):
    df = pd.read_csv(file_path, header=None)

    is_con = all(np.issubdtype(df[col].dtype, np.number) for col in df.iloc[:, :-1].columns)

    for col_idx in df.iloc[:, :-1].columns:
        if np.issubdtype(df[col_idx].dtype, np.number):
            print(f"列 {col_idx} 为连续型")
        else:
            print(f"列 {col_idx} 为离散型")

    if is_con:
        df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
        y = df['species'].values
        X = df.drop('species', axis=1).values
    else:
        df.columns=['class','cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing',
        'gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring',
        'stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type',
        'spore-print-color','population','habitat']
        y = df['class'].values
        X_df = df.drop('class', axis=1)
        le_dict = {}
        for col in X_df.columns:
            le = LabelEncoder()
            X_df[col] = le.fit_transform(X_df[col])
            le_dict[col] = le

            X = X_df.values
    return df, is_con, X, y

import os

def plot_confusion_matrix(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.savefig(os.path.join(save_path, f'{title}.png'))
    plt.close()
    return None

def main():
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    mushroom_path = r'data/mushrooms.csv'

    save_dir = r'outputs/da'
    os.makedirs(save_dir, exist_ok=True)

    datasets = [('Mushrooms', mushroom_path)]
    accuracies = {'Mushrooms': {}}

    for dataset_name, file_path in datasets:
        df, is_con, X, y = read_data(file_path)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 一层决策树
        cart = DecisionTreeClassifier(max_depth=1, random_state=42)
        cart.fit(X_train, y_train)
        cart_pred = cart.predict(X_test)
        cart_acc = accuracy_score(y_test, cart_pred)
        print(f"{dataset_name} 一层决策树 准确率: {cart_acc * 100:.2f}%")
        cart_precision = precision_score(y_test, cart_pred, pos_label='p')
        cart_recall = recall_score(y_test, cart_pred, pos_label='p')
        print(f"{dataset_name} 一层决策树 精确率(毒): {cart_precision * 100:.2f}%")
        print(f"{dataset_name} 一层决策树 召回率(毒): {cart_recall * 100:.2f}%")
        accuracies[dataset_name]['OneLayerCART'] = cart_acc

        # AdaBoost
        base_estimator = DecisionTreeClassifier(max_depth=1, random_state=42)
        ab = AdaBoostClassifier(estimator=base_estimator, n_estimators=10, random_state=42)
        ab.fit(X_train, y_train)
        ab_pred = ab.predict(X_test)
        ab_acc = accuracy_score(y_test, ab_pred)
        print(f"{dataset_name} AdaBoost 准确率: {ab_acc * 100:.2f}%")
        ab_precision = precision_score(y_test, ab_pred, pos_label='p')
        ab_recall = recall_score(y_test, ab_pred, pos_label='p')
        print(f"{dataset_name} AdaBoost 精确率(毒): {ab_precision * 100:.2f}%")
        print(f"{dataset_name} AdaBoost 召回率(毒): {ab_recall * 100:.2f}%")

        train_errors = []
        test_errors = []
        for y_pred_tr, y_pred_te in zip(ab.staged_predict(X_train), ab.staged_predict(X_test)):
            train_errors.append(1.0 - accuracy_score(y_train, y_pred_tr))
            test_errors.append(1.0 - accuracy_score(y_test, y_pred_te))
        plt.figure()
        iters = np.arange(1, len(train_errors)+1)
        plt.plot(iters, train_errors, label='训练误差')
        plt.plot(iters, test_errors, label='测试误差')
        best_idx = int(np.argmin(test_errors))
        best_iter = best_idx + 1
        best_te = float(test_errors[best_idx])
        best_tr = float(train_errors[best_idx])
        plt.axvline(best_iter, color='red', linestyle='--', alpha=0.6, label='最佳迭代')
        plt.scatter([best_iter], [best_te], color='red')
        plt.annotate(f'最佳: {best_iter}, 测试误差 {best_te*100:.2f}%',
                     xy=(best_iter, best_te), xytext=(10, -10), textcoords='offset points',
                     color='red')
        plt.xlabel('迭代次数')
        plt.ylabel('误差')
        plt.title(f'{dataset_name} AdaBoost 学习曲线')
        plt.legend()
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'{dataset_name} AdaBoost 学习曲线.png'), bbox_inches='tight')
        plt.close()
        print(f"{dataset_name} AdaBoost 学习曲线最佳迭代: {best_iter}, 最低测试误差: {best_te*100:.2f}%, 对应训练误差: {best_tr*100:.2f}%")
        accuracies[dataset_name]['AdaBoost'] = ab_acc

        # 可视化混淆矩阵
        plot_confusion_matrix(y_test, cart_pred, f'{dataset_name} 一层决策树 混淆矩阵', save_dir)
        plot_confusion_matrix(y_test, ab_pred, f'{dataset_name} AdaBoost 混淆矩阵', save_dir)

    categories = ['准确率', '精确率', '召回率']
    cart_values = [cart_acc, cart_precision, cart_recall]
    ab_values = [ab_acc, ab_precision, ab_recall]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, cart_values, width, label='一层决策树', color='blue')
    rects2 = ax.bar(x + width/2, ab_values, width, label='AdaBoost', color='green')

    ax.set_ylabel('百分数')
    ax.set_title('mushrooms 数据集算法指标比较')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)

    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height * 100:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords='offset points',
                    ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '准确率比较柱状图.png'))
    plt.close()
    return None

if __name__ == "__main__":
    main()
