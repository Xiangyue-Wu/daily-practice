import pandas as pd
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score
import os

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
        y = df.iloc[:, -1].values
        X = df.iloc[:, :-1].values.astype(float)
        y_unique, y_inv = np.unique(y, return_inverse=True)
        y = np.where(y_inv == 1, 1.0, -1.0)
    else:
        df.columns=['class','cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing',
        'gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring',
        'stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type',
        'spore-print-color','population','habitat']
        y_raw = df.iloc[:, 0].values
        X_raw = df.iloc[:, 1:].values
        X_num = np.zeros_like(X_raw, dtype=float)
        for j in range(X_raw.shape[1]):
            col = X_raw[:, j]
            uniques = np.unique(col)
            mapper = {val: idx for idx, val in enumerate(uniques)}
            X_num[:, j] = np.vectorize(mapper.get)(col).astype(float)
        if 'p' in y_raw:
            y = np.where(y_raw == 'p', 1.0, -1.0)
        else:
            y_unique, y_inv = np.unique(y_raw, return_inverse=True)
            y = np.where(y_inv == 1, 1.0, -1.0)
        X = X_num

    return df, is_con, X, y

def bootstrap_sample(X, y, sample_size):
    indices = np.random.choice(len(X), size=sample_size, replace=True)
    X_sample = X[indices]
    y_sample = y[indices]
    return X_sample, y_sample

def manual_train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    test_count = int(len(X) * test_size)
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    return X_train, X_test, y_train, y_test

def manual_accuracy_score(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    return correct / len(y_true)

class DecisionTreeClassifierWithWeight:
    def __init__(self):
        self.best_err = 1.0
        self.best_fea_id = 0
        self.best_thres = 0.0
        self.best_op = 1

    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones(len(X)) / len(X)
        else:
            sample_weight = sample_weight / np.sum(sample_weight)
        m, n = X.shape
        self.best_err = 1.0
        for i in range(n):
            feature_vals = X[:, i]
            unique_vals = np.unique(feature_vals)
            if len(unique_vals) <= 1:
                continue
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0
            for thres in thresholds:
                for op in (0, 1):
                    if op == 1:
                        y_pred = np.where(feature_vals >= thres, 1.0, -1.0)
                    else:
                        y_pred = np.where(feature_vals < thres, 1.0, -1.0)
                    err = np.sum(sample_weight * (y_pred != y).astype(float))
                    if err < self.best_err:
                        self.best_err = err
                        self.best_fea_id = i
                        self.best_thres = thres
                        self.best_op = op
        return self

    def predict(self, X):
        feature_vals = X[:, self.best_fea_id]
        if self.best_op == 1:
            y_pred = np.where(feature_vals >= self.best_thres, 1.0, -1.0)
        else:
            y_pred = np.where(feature_vals < self.best_thres, 1.0, -1.0)
        return y_pred

class AdaBoostClassifier_:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.estimators = []
        self.alphas = []
        self.sample_weight_history = []

    def fit(self, X, y):
        m = len(X)
        sample_weight = np.ones(m) / m
        for _ in range(self.n_estimators):
            dtc = DecisionTreeClassifierWithWeight().fit(X, y, sample_weight)
            err = dtc.best_err
            err = np.clip(err, 1e-10, 1 - 1e-10)
            alpha = 0.5 * np.log((1.0 - err) / err)
            self.alphas.append(alpha)
            self.estimators.append(dtc)
            y_pred = dtc.predict(X)
            sample_weight *= np.exp(-alpha * y * y_pred)
            sample_weight /= np.sum(sample_weight)
            self.sample_weight_history.append(sample_weight.copy())
        return self

    def predict(self, X):
        m = len(X)
        y_pred_sum = np.zeros(m)
        for estimator, alpha in zip(self.estimators, self.alphas):
            prediction = estimator.predict(X)
            y_pred_sum += alpha * prediction
        final_pred = np.sign(y_pred_sum)
        return final_pred.astype(float)

def plot_confusion_matrix(y_true, y_pred, title, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{title}.png"), bbox_inches='tight')
    plt.close()
    return None

def adaboost_learning_curve(model, X_train, y_train, X_test, y_test):
    train_errors = []
    test_errors = []
    m_tr = len(X_train)
    m_te = len(X_test)
    y_sum_tr = np.zeros(m_tr)
    y_sum_te = np.zeros(m_te)
    for est, alpha in zip(model.estimators, model.alphas):
        y_sum_tr += alpha * est.predict(X_train)
        y_sum_te += alpha * est.predict(X_test)
        pred_tr = np.sign(y_sum_tr)
        pred_te = np.sign(y_sum_te)
        train_errors.append(1.0 - np.mean(pred_tr == y_train))
        test_errors.append(1.0 - np.mean(pred_te == y_test))
    return np.array(train_errors), np.array(test_errors)

def weight_evolution(model, X_train, y_train):
    mis_avg = []
    corr_avg = []
    for est, w in zip(model.estimators, model.sample_weight_history):
        pred = est.predict(X_train)
        mis_mask = (pred != y_train)
        corr_mask = ~mis_mask
        mis_avg.append(np.mean(w[mis_mask]) if np.any(mis_mask) else 0.0)
        corr_avg.append(np.mean(w[corr_mask]) if np.any(corr_mask) else 0.0)
    return np.array(mis_avg), np.array(corr_avg)

def main():
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    mushroom_path = r'data/mushrooms.csv'

    datasets = [('Mushrooms', mushroom_path)]
    accuracies = {'Mushrooms': {}}
    save_dir = r'outputs/ha'

    for dataset_name, file_path in datasets:
        df, is_con, X, y = read_data(file_path)
        X_train, X_test, y_train, y_test = manual_train_test_split(X, y, test_size=0.2, random_state=42)

        # 一层决策树
        cart = DecisionTreeClassifierWithWeight()
        cart.fit(X_train, y_train)
        cart_pred = cart.predict(X_test)
        cart_acc = manual_accuracy_score(y_test, cart_pred)
        print(f"{dataset_name} 一层决策树 准确率: {cart_acc * 100:.2f}%")
        accuracies[dataset_name]['OneLayerCART'] = cart_acc
        cart_precision = precision_score(y_test, cart_pred, pos_label=1.0)
        cart_recall = recall_score(y_test, cart_pred, pos_label=1.0)
        print(f"{dataset_name} 一层决策树 精确率(毒): {cart_precision * 100:.2f}%")
        print(f"{dataset_name} 一层决策树 召回率(毒): {cart_recall * 100:.2f}%")

        # AdaBoost
        ab = AdaBoostClassifier_(n_estimators=50)
        ab.fit(X_train, y_train)
        ab_pred = ab.predict(X_test)
        ab_acc = manual_accuracy_score(y_test, ab_pred)
        print(f"{dataset_name} AdaBoost 准确率: {ab_acc * 100:.2f}%")
        accuracies[dataset_name]['AdaBoost'] = ab_acc
        ab_precision = precision_score(y_test, ab_pred, pos_label=1.0)
        ab_recall = recall_score(y_test, ab_pred, pos_label=1.0)
        print(f"{dataset_name} AdaBoost 精确率(毒): {ab_precision * 100:.2f}%")
        print(f"{dataset_name} AdaBoost 召回率(毒): {ab_recall * 100:.2f}%")

        tr_err, te_err = adaboost_learning_curve(ab, X_train, y_train, X_test, y_test)
        plt.figure()
        iters = np.arange(1, len(tr_err)+1)
        plt.plot(iters, tr_err, label='训练误差')
        plt.plot(iters, te_err, label='测试误差')
        best_idx = int(np.argmin(te_err))
        best_iter = best_idx + 1
        best_te = float(te_err[best_idx])
        best_tr = float(tr_err[best_idx])
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

        mis_avg, corr_avg = weight_evolution(ab, X_train, y_train)
        plt.figure()
        plt.plot(np.arange(1, len(mis_avg)+1), mis_avg, label='误分样本平均权重')
        plt.plot(np.arange(1, len(corr_avg)+1), corr_avg, label='正确样本平均权重')
        plt.xlabel('迭代次数')
        plt.ylabel('权重')
        plt.title(f'{dataset_name} AdaBoost 权重演化')
        plt.legend()
        plt.savefig(os.path.join(save_dir, f'{dataset_name} AdaBoost 权重演化.png'), bbox_inches='tight')
        plt.close()

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
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, 'accuracy_comparison.png'), bbox_inches='tight')
    plt.close()
    return None

if __name__ == "__main__":
    main()
