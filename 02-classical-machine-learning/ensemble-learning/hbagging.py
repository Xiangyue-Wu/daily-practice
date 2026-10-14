import pandas as pd
import numpy as np
import multiprocessing as mp
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import os
import numpy.random as rnd

def read_data(file_path):
    df = pd.read_csv(file_path, header=None)
    df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    y = df.iloc[:, -1].values
    X = df.iloc[:, :-1].values
    return df, True, X, y

def bootstrap_sample(X, y, sample_size, rng=None):
    if rng is None:
        rng = np.random
    indices = rng.choice(len(X), size=sample_size, replace=True)
    X_sample = X[indices]
    y_sample = y[indices]
    return X_sample, y_sample

def manual_accuracy_score(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    return correct / len(y_true)

def test_split(index, value, dataset):
    mask = dataset[:, index] < value
    left = dataset[mask]
    right = dataset[~mask]
    return left, right

def gini_index(groups, classes):
    n_instances = sum(len(group) for group in groups)
    if n_instances == 0:
        return 0.0
    gini = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        counts = np.bincount(group[:, -1].astype(int))
        p = counts / size
        score = np.sum(p ** 2)
        gini += (1.0 - score) * (size / n_instances)
    return gini

def get_split(dataset, n_features):
    class_values = np.unique(dataset[:, -1])
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    gini_before = gini_index([dataset], class_values)
    features = rnd.choice(dataset.shape[1] - 1, n_features, replace=False)
    for index in features:
        unique_values = np.unique(dataset[:, index])
        for value in unique_values:
            groups = test_split(index, value, dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, value, gini, groups
    gain = max(gini_before - b_score, 0.0)
    return {'index': b_index, 'value': b_value, 'groups': b_groups, 'gain': gain, 'n_node_samples': dataset.shape[0]}

def to_terminal(group):
    if len(group) == 0:
        return 0
    return np.bincount(group[:, -1].astype(int)).argmax()

def split(node, max_depth, min_size, n_features, depth):
    left, right = node['groups']
    del(node['groups'])
    if len(left) == 0 or len(right) == 0:
        node['left'] = node['right'] = to_terminal(np.concatenate((left, right)))
        return None
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return None
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left, n_features)
        split(node['left'], max_depth, min_size, n_features, depth + 1)
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right, n_features)
        split(node['right'], max_depth, min_size, n_features, depth + 1)
    return None

def build_tree(train, max_depth, min_size, n_features):
    root = get_split(train, n_features)
    split(root, max_depth, min_size, n_features, 1)
    return root

def predict_row(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict_row(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict_row(node['right'], row)
        else:
            return node['right']

def subsample(dataset, ratio):
    n_sample = int(dataset.shape[0] * ratio)
    indices = rnd.choice(dataset.shape[0], n_sample, replace=True)
    return dataset[indices]

def build_tree_wrapper(args):
    train, max_depth, min_size, n_features, sample_size = args
    sample = subsample(train, sample_size)
    tree = build_tree(sample, max_depth, min_size, n_features)
    return tree

class OneLayerCART:
    def __init__(self, is_con):
        self.feature_index = None
        self.split_value = None
        self.is_con = is_con
        self.left_label = None
        self.right_label = None
        self.classes_ = None
        self.label_encoder = None
        self.left_proba = None
        self.right_proba = None
        self.min_samples_leaf = 1

    def gini(self, y):
        classes = np.unique(y)
        gini = 1.0
        for cls in classes:
            p = np.sum(y == cls) / len(y)
            gini -= p ** 2
        return gini

    def split_data(self, X, y, feature_index, value):
        if self.is_con:
            left = y[X[:, feature_index] <= value]
            right = y[X[:, feature_index] > value]
        else:
            left = y[X[:, feature_index] == value]
            right = y[X[:, feature_index] != value]
        return left, right

    def find_best_split(self, X, y, feature_index, is_con):
        best_gain = -np.inf
        best_value = None

        if is_con:
            vals = np.unique(X[:, feature_index])
            if vals.size >= 2:
                possible_splits = vals[:-1]
            else:
                possible_splits = []
        else:
            possible_splits = np.unique(X[:, feature_index])

        gini_before = self.gini(y)

        for value in possible_splits:
            left, right = self.split_data(X, y, feature_index, value)
            if len(left) < self.min_samples_leaf or len(right) < self.min_samples_leaf:
                continue
            gini_left = self.gini(left)
            gini_right = self.gini(right)
            weighted_gini = (len(left) / len(y)) * gini_left + (len(right) / len(y)) * gini_right
            gain = gini_before - weighted_gini
            if gain > best_gain:
                best_gain = gain
                best_value = value

        return best_gain, best_value, is_con

    def fit(self, X, y):
        best_gain = -np.inf
        for feature_index in range(X.shape[1]):
            gain, value, _ = self.find_best_split(X, y, feature_index, self.is_con)
            if gain > best_gain:
                best_gain = gain
                self.feature_index = feature_index
                self.split_value = value
        left, right = self.split_data(X, y, self.feature_index, self.split_value)
        self.classes_ = np.unique(y)
        left_counts = np.array([(left == c).sum() for c in self.classes_])
        right_counts = np.array([(right == c).sum() for c in self.classes_])
        self.left_label = self.classes_[np.argmax(left_counts)]
        self.right_label = self.classes_[np.argmax(right_counts)]
        self.left_proba = left_counts / len(left) if len(left) > 0 else np.zeros(len(self.classes_))
        self.right_proba = right_counts / len(right) if len(right) > 0 else np.zeros(len(self.classes_))
        return None

    def predict(self, X):
        predictions = []
        for sample in X:
            if self.is_con:
                if sample[self.feature_index] <= self.split_value:
                    predictions.append(self.left_label)
                else:
                    predictions.append(self.right_label)
            else:
                if sample[self.feature_index] == self.split_value:
                    predictions.append(self.left_label)
                else:
                    predictions.append(self.right_label)
        return np.array(predictions)

    def predict_proba(self, X):
        probas = []
        for sample in X:
            if self.is_con:
                if sample[self.feature_index] <= self.split_value:
                    probas.append(self.left_proba)
                else:
                    probas.append(self.right_proba)
            else:
                if sample[self.feature_index] == self.split_value:
                    probas.append(self.left_proba)
                else:
                    probas.append(self.right_proba)
        return np.vstack(probas)

class RandomForest:
    def __init__(self, n_trees=10, is_con=False, subsample_size=1.0, n_features=None, random_state=42, max_depth=10, min_size=1):
        self.n_trees = n_trees
        self.is_con = is_con
        self.subsample_size = subsample_size
        self.n_features = n_features
        self.trees = []
        self.random_state = random_state
        self.classes_ = None
        self.max_depth = max_depth
        self.min_size = min_size

    def fit(self, X, y):
        self.label_encoder = LabelEncoder()
        y_enc = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        dataset = np.column_stack([X, y_enc])
        if self.n_features is None:
            self.n_features = int(np.sqrt(X.shape[1]))
        n_train = dataset.shape[0]
        n_sample = int(n_train * self.subsample_size)
        self.trees = []
        self.inbag_indices = []
        for _ in range(self.n_trees):
            idx_inbag = rnd.choice(n_train, n_sample, replace=True)
            sample = dataset[idx_inbag]
            tree = build_tree(sample, self.max_depth, self.min_size, self.n_features)
            self.trees.append(tree)
            self.inbag_indices.append(set(idx_inbag.tolist()))
        self._compute_oob(dataset)
        self._compute_feature_importances(X.shape[1])
        return None

    def predict(self, X):
        tree_preds = np.array([[predict_row(tree, row) for row in X] for tree in self.trees])
        votes = np.apply_along_axis(lambda x: np.bincount(x.astype(int), minlength=len(self.classes_)), axis=0, arr=tree_preds)
        pred_indices = np.argmax(votes, axis=0)
        return self.label_encoder.inverse_transform(pred_indices)

    def predict_with_tree(self, X, tree):
        return np.array([predict_row(tree, row) for row in X])

    def _compute_oob(self, dataset):
        n_train = dataset.shape[0]
        votes = [[] for _ in range(n_train)]
        for t_idx, tree in enumerate(self.trees):
            inbag = self.inbag_indices[t_idx]
            for j in range(n_train):
                if j not in inbag:
                    pred = predict_row(tree, dataset[j, :-1])
                    votes[j].append(int(pred))
        y_true = dataset[:, -1].astype(int)
        oob_preds = np.full(n_train, -1, dtype=int)
        used = []
        for j in range(n_train):
            if len(votes[j]) > 0:
                counts = np.bincount(np.array(votes[j], dtype=int), minlength=len(self.classes_))
                oob_preds[j] = int(np.argmax(counts))
                used.append(j)
        if len(used) == 0:
            self.oob_score_ = None
            self.oob_predictions_ = None
        else:
            acc = np.mean(y_true[used] == oob_preds[used])
            self.oob_score_ = acc
            self.oob_predictions_ = oob_preds

    def _compute_feature_importances(self, n_features):
        def collect(tree):
            imp = np.zeros(n_features)
            stack = [tree]
            while stack:
                node = stack.pop()
                if isinstance(node, dict):
                    idx = node.get('index', None)
                    gain = node.get('gain', 0.0)
                    weight = node.get('n_node_samples', 0)
                    if idx is not None and idx >= 0:
                        imp[idx] += gain * weight
                    left = node.get('left')
                    right = node.get('right')
                    if isinstance(left, dict):
                        stack.append(left)
                    if isinstance(right, dict):
                        stack.append(right)
            return imp
        total = np.zeros(n_features)
        for tree in self.trees:
            total += collect(tree)
        s = total.sum()
        if s > 0:
            self.feature_importances_ = total / s
        else:
            self.feature_importances_ = total

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

def main():
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    iris_path = r'data/iris.data'
    save_dir = r'outputs/hb'

    datasets = [('Iris', iris_path)]

    for dataset_name, file_path in datasets:
        df, is_con, X, y = read_data(file_path)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        # For CART
        cart_accuracies_fold = []
        cart_y_true_all = []
        cart_y_pred_all = []
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            cart = OneLayerCART(is_con)
            cart.fit(X_train, y_train)
            feature_names = df.columns[:-1].tolist()
            if cart.feature_index is not None:
                print(f"{dataset_name} 一层决策树 使用特征: {feature_names[cart.feature_index]}, 阈值: {cart.split_value}")
            cart_pred = cart.predict(X_test)
            acc = manual_accuracy_score(y_test, cart_pred)
            cart_accuracies_fold.append(acc)
            cart_y_true_all.extend(y_test)
            cart_y_pred_all.extend(cart_pred)
        cart_acc = np.mean(cart_accuracies_fold)
        print(f"{dataset_name} 一层决策树 平均准确率: {cart_acc * 100:.2f}%")
        print(f"{dataset_name} 一层决策树 准确率标准差: {np.std(cart_accuracies_fold) * 100:.2f}%")

        # For RF
        rf_accuracies_fold = []
        rf_y_true_all = []
        rf_y_pred_all = []
        rf_oob_scores_fold = []
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            rf = RandomForest(n_trees=50, is_con=is_con, n_features=2, random_state=57)
            rf.fit(X_train, y_train)
            rf_pred = rf.predict(X_test)
            acc = manual_accuracy_score(y_test, rf_pred)
            rf_accuracies_fold.append(acc)
            rf_y_true_all.extend(y_test)
            rf_y_pred_all.extend(rf_pred)
            if rf.oob_score_ is not None:
                rf_oob_scores_fold.append(rf.oob_score_)
            feature_names = df.columns[:-1].tolist()
            importances = rf.feature_importances_ if hasattr(rf, 'feature_importances_') else np.zeros(len(feature_names))
            order = np.argsort(importances)[::-1]
            top_pairs = [(feature_names[i], importances[i]) for i in order]
            print(f"{dataset_name} 随机森林 OOB 准确率: {rf.oob_score_ * 100:.2f}%" if rf.oob_score_ is not None else f"{dataset_name} 随机森林 OOB 准确率: 不可用")
            print("随机森林 特征重要性:", [f"{name}:{imp:.3f}" for name, imp in top_pairs])
        rf_acc = np.mean(rf_accuracies_fold)
        print(f"{dataset_name} 随机森林 平均准确率: {rf_acc * 100:.2f}%")
        print(f"{dataset_name} 随机森林 准确率标准差: {np.std(rf_accuracies_fold) * 100:.2f}%")
        if len(rf_oob_scores_fold) > 0:
            print(f"{dataset_name} 随机森林 平均OOB准确率: {np.mean(rf_oob_scores_fold) * 100:.2f}%")
            print(f"{dataset_name} 随机森林 OOB准确率标准差: {np.std(rf_oob_scores_fold) * 100:.2f}%")

        # Plot confusion matrices using aggregated predictions
        plot_confusion_matrix(np.array(cart_y_true_all), np.array(cart_y_pred_all), f'{dataset_name} 一层决策树 混淆矩阵 (5-fold CV)', save_dir)
        plot_confusion_matrix(np.array(rf_y_true_all), np.array(rf_y_pred_all), f'{dataset_name} 随机森林 混淆矩阵 (5-fold CV)', save_dir)

        cart_acc_overall = manual_accuracy_score(np.array(cart_y_true_all), np.array(cart_y_pred_all))
        rf_acc_overall = manual_accuracy_score(np.array(rf_y_true_all), np.array(rf_y_pred_all))
        cart_std = np.std(cart_accuracies_fold)
        rf_std = np.std(rf_accuracies_fold)

        categories = ['准确率', '准确率均值', '准确率标准差']
        cart_values = [cart_acc_overall, cart_acc, cart_std]
        rf_values = [rf_acc_overall, rf_acc, rf_std]

        x = np.arange(len(categories))
        width = 0.35

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width / 2, cart_values, width, label='一层决策树', color='blue')
        rects2 = ax.bar(x + width / 2, rf_values, width, label='随机森林', color='green')

        ax.set_ylabel('百分数')
        ax.set_title(f'{dataset_name} 数据集算法指标比较 (5-fold CV)')
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
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f'{dataset_name}_accuracy_comparison_cv.png'), bbox_inches='tight')
        plt.close()

    return None

if __name__ == "__main__":
    main()
