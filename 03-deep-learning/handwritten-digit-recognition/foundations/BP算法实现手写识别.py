import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

digits = load_digits()
X = digits.data
y = digits.target

V = np.random.random((64, 100)) * 2 - 1
W = np.random.random((100, 10)) * 2 - 1

X_train, X_test, y_train, y_test = train_test_split(X, y)
label_train = LabelBinarizer().fit_transform(y_train)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dsigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s * (1 - s)


def train(X, y, Steps=10000, lr=0.12):
    global W, V
    accuracies = []  # 用于存储每个步骤的准确率
    for n in range(Steps):
        i = np.random.randint(X.shape[0])
        x = X[i]
        x = np.atleast_2d(x)
        L1 = sigmoid(np.dot(x, V))
        L2 = sigmoid(np.dot(L1, W))
        L2_delta = (y[i] - L2) * dsigmoid(np.dot(L1, W))
        L1_delta = np.dot(L2_delta, W.T) * dsigmoid(np.dot(x, V))
        W += lr * np.dot(L1.T, L2_delta)
        V += lr * np.dot(x.T, L1_delta)
        if n % 1000 == 0:
            output = predict(X_test)
            predictions = np.argmax(output, axis=1)
            acc = np.mean(np.equal(predictions, y_test))
            accuracies.append(acc)  # 将准确率添加到列表中
            print("steps:", n, "accuracy:", acc)
    plt.plot(accuracies)  # 绘制最终的准确率曲线
    plt.show()  # 显示图像
    plt.pause(0.1)  # 暂停0.1秒，以便手动关闭图像窗口并继续执行代码，你也可以调整这个时间来满足你的需要。


def predict(x):
    L1 = sigmoid(np.dot(x, V))
    L2 = sigmoid(np.dot(L1, W))
    return L2


train(X_train, label_train, 20000)