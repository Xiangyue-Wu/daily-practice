
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression


# ==================== 逻辑回归 ====================

def build_logistic_regression():
    return LogisticRegression(max_iter=1000)


# ==================== TextCNN ====================

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, num_classes=2):
        super().__init__()

        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # 多尺度卷积：3、4、5
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 100, (k, embed_dim))
            for k in [3, 4, 5]
        ])

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(300, num_classes)

    def forward(self, x):
        # [batch, length] -> [batch, 1, length, embed_dim]
        x = self.embedding(x).unsqueeze(1)

        # 卷积与激活
        x = [
            torch.relu(conv(x)).squeeze(3)
            for conv in self.convs
        ]

        # 全局最大池化
        x = [
            torch.max_pool1d(feature, feature.size(2)).squeeze(2)
            for feature in x
        ]

        # 特征拼接与分类
        x = torch.cat(x, dim=1)
        x = self.dropout(x)

        return self.fc(x)


# ==================== Bi-LSTM ====================

class TextLSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=100,
        hidden_dim=64,
        num_classes=2
    ):
        super().__init__()

        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # 双向 LSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # 双向隐藏状态拼接后分类
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)

        # 优化 LSTM 参数的内存布局
        self.lstm.flatten_parameters()

        _, (h_n, _) = self.lstm(x)

        # 拼接前向与后向隐藏状态
        hidden = torch.cat(
            (h_n[-2, :, :], h_n[-1, :, :]),
            dim=1
        )

        return self.fc(hidden)


# ==================== 独立测试 ====================

def test_models():
    vocab_size = 5000
    batch_size = 4
    seq_len = 100

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # 模拟文本索引输入
    x = torch.randint(
        0, vocab_size, (batch_size, seq_len)
    ).to(device)

    # 初始化模型
    lr_model = build_logistic_regression()
    cnn_model = TextCNN(vocab_size).to(device)
    lstm_model = TextLSTM(vocab_size).to(device)

    cnn_model.eval()
    lstm_model.eval()

    # 检查输出维度
    with torch.no_grad():
        cnn_output = cnn_model(x)
        lstm_output = lstm_model(x)

    print(f"运行设备：{device}")
    print(f"逻辑回归初始化：{type(lr_model).__name__}")
    print(f"TextCNN 输出维度：{tuple(cnn_output.shape)}")
    print(f"Bi-LSTM 输出维度：{tuple(lstm_output.shape)}")

    assert cnn_output.shape == (batch_size, 2)
    assert lstm_output.shape == (batch_size, 2)

    print("所有模型测试通过！")


if __name__ == "__main__":
    test_models()
