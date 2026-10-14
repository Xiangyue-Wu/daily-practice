import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import os

def read_data(file_path):
    df = pd.read_csv(file_path, header=None)
    df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    y = df['species'].values
    X = df.drop('species', axis=1).values
    return df, True, X, y

 

def main():
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    iris_path = r'data/iris.data'
    save_dir = r'outputs/db'

    datasets = [('Iris', iris_path)]

    for dataset_name, file_path in datasets:
        df, is_con, X, y = read_data(file_path)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        cart = DecisionTreeClassifier(max_depth=1, random_state=42)
        cart.fit(X_train, y_train)
        cart_pred = cart.predict(X_test)
        cart_acc = accuracy_score(y_test, cart_pred)
        print(f"{dataset_name} 一层决策树 准确率: {cart_acc * 100:.2f}%")
        root_idx = cart.tree_.feature[0]
        threshold = cart.tree_.threshold[0]
        feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        print(f"{dataset_name} 一层决策树 使用特征: {feature_names[root_idx]}, 阈值: {threshold}")
        cart_importances = cart.feature_importances_
        order_cart = np.argsort(cart_importances)[::-1]
        print("一层决策树 特征重要性:", [f"{feature_names[i]}:{cart_importances[i]:.3f}" for i in order_cart])

        rf = RandomForestClassifier(n_estimators=10, max_depth=1, random_state=42, bootstrap=True, oob_score=True)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_acc = accuracy_score(y_test, rf_pred)
        print(f"{dataset_name} 随机森林 准确率: {rf_acc * 100:.2f}%")
        if hasattr(rf, 'oob_score_') and rf.oob_score_ is not None:
            print(f"{dataset_name} 随机森林 OOB 准确率: {rf.oob_score_ * 100:.2f}%")
        rf_importances = rf.feature_importances_
        order_rf = np.argsort(rf_importances)[::-1]
        print("随机森林 特征重要性:", [f"{feature_names[i]}:{rf_importances[i]:.3f}" for i in order_rf])

        n_repeats = 10
        cart_scores = []
        rf_scores = []
        rf_oob_scores = []
        for i in range(n_repeats):
            X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y, test_size=0.2, random_state=42 + i)
            cart_r = DecisionTreeClassifier(max_depth=1, random_state=42 + i)
            cart_r.fit(X_train_r, y_train_r)
            cart_pred_r = cart_r.predict(X_test_r)
            cart_scores.append(accuracy_score(y_test_r, cart_pred_r))

            rf_r = RandomForestClassifier(n_estimators=10, max_depth=1, random_state=42 + i, bootstrap=True, oob_score=True)
            rf_r.fit(X_train_r, y_train_r)
            rf_pred_r = rf_r.predict(X_test_r)
            rf_scores.append(accuracy_score(y_test_r, rf_pred_r))
            if hasattr(rf_r, 'oob_score_') and rf_r.oob_score_ is not None:
                rf_oob_scores.append(rf_r.oob_score_)

        print(f"{dataset_name} 一层决策树 准确率均值: {np.mean(cart_scores) * 100:.2f}%")
        print(f"{dataset_name} 一层决策树 准确率标准差: {np.std(cart_scores) * 100:.2f}%")
        print(f"{dataset_name} 随机森林 准确率均值: {np.mean(rf_scores) * 100:.2f}%")
        print(f"{dataset_name} 随机森林 准确率标准差: {np.std(rf_scores) * 100:.2f}%")
        if len(rf_oob_scores) > 0:
            print(f"{dataset_name} 随机森林 OOB准确率均值: {np.mean(rf_oob_scores) * 100:.2f}%")
            print(f"{dataset_name} 随机森林 OOB准确率标准差: {np.std(rf_oob_scores) * 100:.2f}%")

        cm1 = confusion_matrix(y_test, cart_pred)
        labels1 = np.unique(np.concatenate([y_test, cart_pred]))
        disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=labels1)
        disp1.plot(cmap=plt.cm.Blues)
        plt.title(f'{dataset_name} 一层决策树 混淆矩阵')
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{dataset_name} 一层决策树 混淆矩阵.png"), bbox_inches='tight')
        plt.close()

        cm2 = confusion_matrix(y_test, rf_pred)
        labels2 = np.unique(np.concatenate([y_test, rf_pred]))
        disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=labels2)
        disp2.plot(cmap=plt.cm.Blues)
        plt.title(f'{dataset_name} 随机森林 混淆矩阵')
        plt.savefig(os.path.join(save_dir, f"{dataset_name} 随机森林 混淆矩阵.png"), bbox_inches='tight')
        plt.close()

        categories = ['准确率', '准确率均值', '准确率标准差']
        cart_values = [cart_acc, np.mean(cart_scores), np.std(cart_scores)]
        rf_values = [rf_acc, np.mean(rf_scores), np.std(rf_scores)]

        x = np.arange(len(categories))
        width = 0.35

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width / 2, cart_values, width, label='一层决策树', color='blue')
        rects2 = ax.bar(x + width / 2, rf_values, width, label='随机森林', color='green')

        ax.set_ylabel('百分数')
        ax.set_title(f'{dataset_name} 数据集算法指标比较')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)

        for rect in list(rects1) + list(rects2):
            height = rect.get_height()
            ax.annotate(f'{height * 100:.2f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords='offset points',
                        ha='center', va='bottom')

        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, f'{dataset_name}_accuracy_comparison.png'), bbox_inches='tight')
        plt.close()

    return None

if __name__ == "__main__":
    main()
