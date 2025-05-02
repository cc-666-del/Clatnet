import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, auc, classification_report, confusion_matrix
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

# 划分训练集和验证集
X_train_seq, X_val_seq, X_train_tfidf, X_val_tfidf, X_train_static, X_val_static, X_train_time, X_val_time, y_train, y_val = train_test_split(
    X_train_seq, X_train_tfidf, X_train_static, X_train_time, y_train, test_size=0.1, random_state=42
)

# 创建 DataLoader
batch_size = 64
train_dataset = TensorDataset(X_train_seq, X_train_tfidf, X_train_static, X_train_time, y_train)
val_dataset = TensorDataset(X_val_seq, X_val_tfidf, X_val_static, X_val_time, y_val)
test_dataset = TensorDataset(X_test_seq, X_test_tfidf, X_test_static, X_test_time, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义模型
class CNN_LSTM_Attention(nn.Module):
    def __init__(self, input_dim_seq, input_dim_tfidf, input_dim_static, input_dim_time, cnn_channels=32, lstm_hidden=32,
                 attention_heads=2, attention_hidden=32, output_dim=1, dropout_prob=0.4):
        super(CNN_LSTM_Attention, self).__init__()
        self.conv1 = nn.Conv1d(2, cnn_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(cnn_channels)
        self.bn2 = nn.BatchNorm1d(cnn_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.AdaptiveMaxPool1d(4)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_prob)

        self.lstm = nn.LSTM(input_dim_tfidf, lstm_hidden, batch_first=True, num_layers=2, bidirectional=True, dropout=0.3)
        self.attention = nn.MultiheadAttention(embed_dim=(lstm_hidden * 2) + cnn_channels, num_heads=attention_heads)
        self.time_attention_fc = nn.Linear(input_dim_time, 64)
        self.time_attention = nn.MultiheadAttention(embed_dim=64, num_heads=4)
        self.fc_static = nn.Linear(input_dim_static, attention_hidden)
        self.fc_final = nn.Sequential(
            nn.Linear((lstm_hidden * 2) + cnn_channels + attention_hidden + 64, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, output_dim)
        )

    def forward(self, seq_data, tfidf_data, static_data, time_data):
        seq_data = seq_data.permute(0, 2, 1)
        seq_data = self.relu(self.bn1(self.conv1(seq_data)))
        seq_data = self.maxpool(self.relu(self.bn2(self.conv2(seq_data))))
        seq_data = self.global_avg_pool(seq_data).squeeze(-1)
        seq_data = self.dropout(seq_data)

        lstm_output, _ = self.lstm(tfidf_data.unsqueeze(1))
        lstm_output = lstm_output[:, -1, :]
        lstm_output = self.dropout(lstm_output)

        combined_data = torch.cat((seq_data, lstm_output), dim=1).unsqueeze(0)
        attn_output, _ = self.attention(combined_data, combined_data, combined_data)
        attn_output = attn_output.squeeze(0)

        static_output = self.relu(self.fc_static(static_data))

        time_data = time_data.unsqueeze(-1)
        time_attn_input = self.relu(self.time_attention_fc(time_data))
        time_attn_input = time_attn_input.permute(1, 0, 2)
        time_attn_output, _ = self.time_attention(time_attn_input, time_attn_input, time_attn_input)
        time_attn_output = time_attn_output.mean(dim=0)

        combined_all = torch.cat((attn_output, static_output, time_attn_output), dim=1)
        combined_all = self.dropout(combined_all)
        output = self.fc_final(combined_all)

        return output

# 获取输入特征的维度
input_dim_seq = X_train_seq.shape[2]
input_dim_tfidf = X_train_tfidf.shape[1]
input_dim_static = X_train_static.shape[1]
input_dim_time = 1

model = CNN_LSTM_Attention(
    input_dim_seq=input_dim_seq,
    input_dim_tfidf=input_dim_tfidf,
    input_dim_static=input_dim_static,
    input_dim_time=input_dim_time,
    cnn_channels=32,
    lstm_hidden=32,
    attention_heads=2,
    attention_hidden=32,
    output_dim=1,
    dropout_prob=0.4
)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 定义损失函数和优化器
pos_weight = torch.tensor([2.0], device=device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

# 修改训练函数以记录验证集 Loss 和 AUC
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=20):
    train_losses = []
    val_losses = []
    train_aucs = []
    val_aucs = []

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        all_labels = []
        all_probs = []

        for seq_inputs, tfidf_inputs, static_inputs, time_inputs, labels in train_loader:
            seq_inputs = seq_inputs.to(device)
            tfidf_inputs = tfidf_inputs.to(device)
            static_inputs = static_inputs.to(device)
            time_inputs = time_inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(seq_inputs, tfidf_inputs, static_inputs, time_inputs)
            loss = criterion(outputs.view(-1), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)

        train_losses.append(running_loss / len(train_loader))
        train_aucs.append(roc_auc_score(np.array(all_labels), np.array(all_probs).flatten()))

        # 验证阶段
        model.eval()
        val_loss = 0.0
        all_val_labels = []
        all_val_probs = []

        with torch.no_grad():
            for seq_inputs, tfidf_inputs, static_inputs, time_inputs, labels in val_loader:
                seq_inputs = seq_inputs.to(device)
                tfidf_inputs = tfidf_inputs.to(device)
                static_inputs = static_inputs.to(device)
                time_inputs = time_inputs.to(device)
                labels = labels.to(device)

                outputs = model(seq_inputs, tfidf_inputs, static_inputs, time_inputs)
                loss = criterion(outputs.view(-1), labels)
                val_loss += loss.item()

                probs = torch.sigmoid(outputs).cpu().numpy()
                all_val_labels.extend(labels.cpu().numpy())
                all_val_probs.extend(probs)

        val_losses.append(val_loss / len(val_loader))
        val_aucs.append(roc_auc_score(np.array(all_val_labels), np.array(all_val_probs).flatten()))

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Train AUC: {train_aucs[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val AUC: {val_aucs[-1]:.4f}")
        scheduler.step(val_losses[-1])

    return train_losses, train_aucs, val_losses, val_aucs

# 训练模型
train_losses, train_aucs, val_losses, val_aucs = train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10
)

# 绘制 Loss 和 AUC 曲线
epochs = range(1, len(train_losses) + 1)

# Loss 曲线
plt.figure()
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.gca().set_facecolor('#FFFACD')  # 浅黄色背景
plt.gcf().set_facecolor('#FFECB3')  # 浅杏黄色图框背景
plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# AUC 曲线
plt.figure()
plt.plot(epochs, train_aucs, label='Train AUC')
plt.plot(epochs, val_aucs, label='Validation AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.title('AUC Curve')
plt.legend()
# 设置背景为黄色系
plt.gca().set_facecolor('#FFFACD')  # 浅黄色背景
plt.gcf().set_facecolor('#FFECB3')  # 浅杏黄色图框背景
plt.savefig('auc_curve.png', dpi=300, bbox_inches='tight')
plt.show()

from sklearn.metrics import confusion_matrix, roc_curve

def evaluate_model(model, test_loader):
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for seq_inputs, tfidf_inputs, static_inputs, time_inputs, labels in test_loader:
            seq_inputs = seq_inputs.to(device)
            tfidf_inputs = tfidf_inputs.to(device)
            static_inputs = static_inputs.to(device)
            time_inputs = time_inputs.to(device)
            labels = labels.to(device)
            outputs = model(seq_inputs, tfidf_inputs, static_inputs, time_inputs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs).flatten()
    roc_auc = roc_auc_score(all_labels, all_probs)

    print(f'Final ROC AUC: {roc_auc:.4f}')
    precision_vals, recall_vals, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall_vals, precision_vals)
    print(f'Final PR AUC: {pr_auc:.4f}')

    # 绘制ROC曲线
    def plot_roc_curve(all_labels, all_probs, roc_auc):
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.4f}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='best')
        plt.gca().set_facecolor('#FFFACD')  # 浅黄色背景
        plt.gcf().set_facecolor('#FFECB3')  # 浅杏黄色图框背景
        plt.savefig('roc_curve_yellow.png', dpi=300, bbox_inches='tight')
        plt.show()

    plot_roc_curve(all_labels, all_probs, roc_auc)

    # 绘制混淆矩阵
    def plot_confusion_matrix(all_labels, all_preds):
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="YlOrBr",
            xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"],
            annot_kws = {"size": 16}  # 调整字体大小
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.gca().set_facecolor('#FFFACD')  # 浅黄色背景
        plt.gcf().set_facecolor('#FFECB3')  # 浅杏黄色图框背景
        plt.savefig('confusion_matrix_yellow.png', dpi=300, bbox_inches='tight')
        plt.show()

    all_preds = (all_probs > 0.5).astype(int)
    plot_confusion_matrix(all_labels, all_preds)

    return roc_auc
# 调用评估函数
final_test_auc = evaluate_model(model, test_loader)
