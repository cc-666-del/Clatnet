import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# 加载训练和测试数据
train_data = torch.load('/home/sun/LHengchang/paper/data/DL/train_data2.pth')
test_data = torch.load('/home/sun/LHengchang/paper/data/DL/test_data2.pth')

X_train_seq = train_data['X_train_seq']
X_train_tfidf = train_data['X_train_tfidf']
X_train_static = train_data['X_train_static']
X_train_time = train_data['X_train_time']
y_train = train_data['y_train']

X_test_seq = test_data['X_test_seq']
X_test_tfidf = test_data['X_test_tfidf']
X_test_static = test_data['X_test_static']
X_test_time = test_data['X_test_time']
y_test = test_data['y_test']

# 加载 ICD 映射（可选，如果需要）
with open('/home/sun/LHengchang/paper/data/DL/icd_mapping.pkl', 'rb') as f:
    mapping_data = pickle.load(f)
    icd_mapping = mapping_data['icd_mapping']
    idx_to_icd = mapping_data['idx_to_icd']

# 数据预处理：将序列特征和时间间隔分离
def prepare_t_lstm_input(seq_data):
    # seq_data 的形状为 [batch_size, seq_len, 2]，其中 2 表示 ICD 编码和相对天数
    # 将其分离为特征和时间间隔
    features = seq_data[:, :, 0].unsqueeze(-1)  # ICD 编码，形状为 [batch_size, seq_len, 1]
    time_gaps = seq_data[:, :, 1]  # 相对天数，形状为 [batch_size, seq_len]

    # 计算时间间隔 delta_t
    delta_t = torch.zeros_like(time_gaps)
    delta_t[:, 1:] = time_gaps[:, 1:] - time_gaps[:, :-1]
    delta_t[:, 0] = 0  # 第一个时间步的时间间隔设为 0

    return features, delta_t

# 准备训练和测试数据
X_train_features, X_train_delta_t = prepare_t_lstm_input(X_train_seq)
X_test_features, X_test_delta_t = prepare_t_lstm_input(X_test_seq)

# 创建 DataLoader
batch_size = 64
train_dataset = TensorDataset(X_train_features, X_train_delta_t, X_train_tfidf, X_train_static, y_train)
test_dataset = TensorDataset(X_test_features, X_test_delta_t, X_test_tfidf, X_test_static, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义 T-LSTM 单元
class TLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 输入门、遗忘门、输出门和候选细胞状态的线性层
        self.x2h = nn.Linear(input_size, 4 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size)

        # 时间衰减系数
        self.decay = nn.Linear(1, hidden_size)

    def forward(self, x_t, h_prev, c_prev, delta_t):
        # 时间衰减
        gamma = torch.exp(-torch.relu(self.decay(delta_t.unsqueeze(-1))))  # [batch_size, hidden_size]
        c_prev = c_prev * gamma

        # LSTM 计算
        gates = self.x2h(x_t) + self.h2h(h_prev)
        i_t, f_t, o_t, g_t = gates.chunk(4, 1)

        i_t = torch.sigmoid(i_t)
        f_t = torch.sigmoid(f_t)
        o_t = torch.sigmoid(o_t)
        g_t = torch.tanh(g_t)

        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t

# 定义 T-LSTM 模型
class TLSTMModel(nn.Module):
    def __init__(self, input_dim_seq, hidden_dim_seq, input_dim_tfidf, input_dim_static, output_dim=1, dropout_prob=0.4):
        super(TLSTMModel, self).__init__()

        self.hidden_dim_seq = hidden_dim_seq

        # T-LSTM 单元
        self.tlstm_cell = TLSTMCell(input_size=input_dim_seq, hidden_size=hidden_dim_seq)
        self.dropout = nn.Dropout(dropout_prob)

        # 全连接层处理 TF-IDF 特征
        self.fc_tfidf = nn.Linear(input_dim_tfidf, 128)
        self.relu = nn.ReLU()

        # 全连接层处理静态特征
        self.fc_static = nn.Linear(input_dim_static, 64)

        # 最终的全连接层
        self.fc_final = nn.Sequential(
            nn.Linear(hidden_dim_seq + 128 + 64, 64),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(64, output_dim)
        )

    def forward(self, seq_inputs, delta_t_inputs, tfidf_input, static_input):
        batch_size, seq_len, _ = seq_inputs.size()
        h_t = torch.zeros(batch_size, self.hidden_dim_seq).to(seq_inputs.device)
        c_t = torch.zeros(batch_size, self.hidden_dim_seq).to(seq_inputs.device)

        for t in range(seq_len):
            x_t = seq_inputs[:, t, :]  # 当前时间步的输入，形状为 [batch_size, input_dim_seq]
            delta_t = delta_t_inputs[:, t]  # 当前时间步的时间间隔，形状为 [batch_size]
            h_t, c_t = self.tlstm_cell(x_t, h_t, c_t, delta_t)

        lstm_out = h_t  # 最后的隐状态作为序列的表示

        # 处理 TF-IDF 特征
        tfidf_out = self.relu(self.fc_tfidf(tfidf_input))

        # 处理静态特征
        static_out = self.relu(self.fc_static(static_input))

        # 拼接所有特征
        combined_features = torch.cat([lstm_out, tfidf_out, static_out], dim=1)
        combined_features = self.dropout(combined_features)

        # 最终的全连接层输出
        output = self.fc_final(combined_features)

        return output

# 获取输入特征的维度
input_dim_seq = X_train_features.shape[2]  # 序列输入的特征维度，应该是 1
input_dim_tfidf = X_train_tfidf.shape[1]
input_dim_static = X_train_static.shape[1]

# 初始化模型
model = TLSTMModel(
    input_dim_seq=input_dim_seq,
    hidden_dim_seq=64,
    input_dim_tfidf=input_dim_tfidf,
    input_dim_static=input_dim_static,
    output_dim=1,
    dropout_prob=0.4
)

# 将模型移至CUDA
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 定义损失函数和优化器
# 计算正负样本数量
num_positive = (y_train == 1).sum().item()
num_negative = (y_train == 0).sum().item()

# 避免除以零的错误
if num_positive == 0:
    pos_weight_value = 1.0  # 或者设置一个默认值
else:
    pos_weight_value = num_negative / num_positive

pos_weight = torch.tensor([pos_weight_value], device=device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 设置学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

# 训练模型
def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=20):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for seq_inputs, delta_t_inputs, tfidf_inputs, static_inputs, labels in train_loader:
            seq_inputs = seq_inputs.to(device)
            delta_t_inputs = delta_t_inputs.to(device)
            tfidf_inputs = tfidf_inputs.to(device)
            static_inputs = static_inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(seq_inputs, delta_t_inputs, tfidf_inputs, static_inputs)
            loss = criterion(outputs.view(-1), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        average_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}")
        scheduler.step(average_loss)

# 评估模型
def evaluate_model(model, test_loader):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for seq_inputs, delta_t_inputs, tfidf_inputs, static_inputs, labels in test_loader:
            seq_inputs = seq_inputs.to(device)
            delta_t_inputs = delta_t_inputs.to(device)
            tfidf_inputs = tfidf_inputs.to(device)
            static_inputs = static_inputs.to(device)
            labels = labels.to(device)
            outputs = model(seq_inputs, delta_t_inputs, tfidf_inputs, static_inputs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            predicted = (probs > 0.5).astype(float)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted)
            all_probs.extend(probs)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs).flatten()

    # 保存真实标签和预测概率
    np.save('tlstm_y_test.npy', all_labels)
    np.save('tlstm_y_pred_proba.npy', all_probs)

    print("Classification Report:")
    print(classification_report(all_labels, all_preds))

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    roc_auc = roc_auc_score(all_labels, all_probs)
    print(f'ROC AUC: {roc_auc:.4f}')

    precision_vals, recall_vals, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall_vals, precision_vals)
    print(f'PR AUC: {pr_auc:.4f}')

    # 绘制并保存PR曲线
    plt.figure()
    plt.plot(recall_vals, precision_vals, label=f'PR AUC = {pr_auc:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.savefig('pr_curve_tlstm.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 绘制并保存混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.title('Confusion Matrix - T-LSTM Model')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix_tlstm.png', dpi=300, bbox_inches='tight')
    plt.show()

# 开始训练和评估
train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=10)
evaluate_model(model, test_loader)
