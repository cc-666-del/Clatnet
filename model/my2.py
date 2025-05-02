import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, auc, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
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

# 加载 ICD 映射
with open('/home/sun/LHengchang/paper/data/DL/icd_mapping.pkl', 'rb') as f:
    mapping_data = pickle.load(f)
    icd_mapping = mapping_data['icd_mapping']
    idx_to_icd = mapping_data['idx_to_icd']

# 创建 DataLoader
batch_size = 64
train_dataset = TensorDataset(X_train_seq, X_train_tfidf, X_train_static, X_train_time, y_train)
test_dataset = TensorDataset(X_test_seq, X_test_tfidf, X_test_static, X_test_time, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义模型
class CNN_LSTM_Attention(nn.Module):
    def __init__(self, input_dim_seq, input_dim_tfidf, input_dim_static, input_dim_time, cnn_channels=32, lstm_hidden=32,
                 attention_heads=2, attention_hidden=32, output_dim=1, dropout_prob=0.4):
        super(CNN_LSTM_Attention, self).__init__()

        # Enhanced CNN with Temporal Feature Fusion
        self.conv1 = nn.Conv1d(2, cnn_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(cnn_channels)
        self.bn2 = nn.BatchNorm1d(cnn_channels)
        self.relu = nn.ReLU()

        # Adjusting to use AdaptiveMaxPool1d
        self.maxpool = nn.AdaptiveMaxPool1d(4)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_prob)

        # LSTM for TF-IDF encoded ICD data
        self.lstm = nn.LSTM(input_dim_tfidf, lstm_hidden, batch_first=True, num_layers=2, bidirectional=True, dropout=0.3)

        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=(lstm_hidden * 2) + cnn_channels, num_heads=attention_heads)

        # Time Attention mechanism
        self.time_attention_fc = nn.Linear(input_dim_time, 64)  # input_dim_time 应为 1
        self.time_attention = nn.MultiheadAttention(embed_dim=64, num_heads=4)

        # Fully connected layers for static data and final output
        self.fc_static = nn.Linear(input_dim_static, attention_hidden)
        self.fc_final = nn.Sequential(
            nn.Linear((lstm_hidden * 2) + cnn_channels + attention_hidden + 64, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, output_dim)
        )

    def forward(self, seq_data, tfidf_data, static_data, time_data):
        # CNN for ICD sequence data
        seq_data = seq_data.permute(0, 2, 1)  # [batch_size, 2, seq_len]
        seq_data = self.relu(self.bn1(self.conv1(seq_data)))
        seq_data = self.maxpool(self.relu(self.bn2(self.conv2(seq_data))))
        seq_data = self.global_avg_pool(seq_data).squeeze(-1)
        seq_data = self.dropout(seq_data)

        # LSTM for TF-IDF encoded ICD data
        lstm_output, _ = self.lstm(tfidf_data.unsqueeze(1))
        lstm_output = lstm_output[:, -1, :]
        lstm_output = self.dropout(lstm_output)

        # Attention mechanism
        combined_data = torch.cat((seq_data, lstm_output), dim=1).unsqueeze(0)
        attn_output, _ = self.attention(combined_data, combined_data, combined_data)
        attn_output = attn_output.squeeze(0)

        # Static data processing
        static_output = self.relu(self.fc_static(static_data))

        # Time data attention
        time_data = time_data.unsqueeze(-1)  # [batch_size, seq_len, 1]
        time_attn_input = self.relu(self.time_attention_fc(time_data))  # [batch_size, seq_len, 64]
        time_attn_input = time_attn_input.permute(1, 0, 2)  # [seq_len, batch_size, 64]
        time_attn_output, _ = self.time_attention(time_attn_input, time_attn_input, time_attn_input)
        time_attn_output = time_attn_output.mean(dim=0)  # [batch_size, 64]

        # Final fully connected layers
        combined_all = torch.cat((attn_output, static_output, time_attn_output), dim=1)
        combined_all = self.dropout(combined_all)
        output = self.fc_final(combined_all)

        return output

# 获取输入特征的维度
input_dim_seq = X_train_seq.shape[2]  # 应为 2
input_dim_tfidf = X_train_tfidf.shape[1]
input_dim_static = X_train_static.shape[1]
input_dim_time = 1  # 每个时间步的特征维度为 1

# 初始化模型
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

# 将模型移至CUDA
device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)

# 定义损失函数和优化器
pos_weight = torch.tensor([2.0], device=device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 设置学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

# 训练模型函数
def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=20):
    train_losses = []
    train_aucs = []

    for epoch in range(num_epochs):
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

            # 记录真实标签和预测概率
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)

        # 计算当前 epoch 的平均 Loss 和 AUC
        average_loss = running_loss / len(train_loader)
        train_losses.append(average_loss)

        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs).flatten()
        train_auc = roc_auc_score(all_labels, all_probs)
        train_aucs.append(train_auc)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}, AUC: {train_auc:.4f}")
        scheduler.step(average_loss)

    return train_losses, train_aucs

# 评估模型函数
def evaluate_model(model, test_loader):
    model.eval()
    all_labels = []
    all_probs = []
    all_preds = []
    with torch.no_grad():
        for seq_inputs, tfidf_inputs, static_inputs, time_inputs, labels in test_loader:
            seq_inputs = seq_inputs.to(device)
            tfidf_inputs = tfidf_inputs.to(device)
            static_inputs = static_inputs.to(device)
            time_inputs = time_inputs.to(device)
            labels = labels.to(device)
            outputs = model(seq_inputs, tfidf_inputs, static_inputs, time_inputs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            predicted = (probs > 0.5).astype(float)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted)
            all_probs.extend(probs)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs).flatten()
    roc_auc = roc_auc_score(all_labels, all_probs)

    print(f'Final ROC AUC: {roc_auc:.4f}')
    precision_vals, recall_vals, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall_vals, precision_vals)
    print(f'Final PR AUC: {pr_auc:.4f}')

    # 绘制PR曲线
    plt.figure()
    plt.plot(recall_vals, precision_vals, label=f'PR AUC = {pr_auc:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.savefig('pr_curve_cnn_lstm_attention_test.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 绘制ROC曲线
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.4f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    plt.savefig('roc_curve_cnn_lstm_attention_test.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 绘制并保存混淆矩阵

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    plt.title('Confusion Matrix - CLATNet Model')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix_cnn_lstm_attention.png', dpi=300, bbox_inches='tight')
    plt.show()

    return roc_auc

# 训练模型
train_losses, train_aucs = train_model(
    model, train_loader, criterion, optimizer, scheduler, num_epochs=15
)

# 绘制 Loss 和 AUC 曲线
epochs = range(1, len(train_losses) + 1)

# Loss 曲线
plt.figure()
plt.plot(epochs, train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend(loc='best')
plt.savefig('training_loss_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# AUC 曲线
plt.figure()
plt.plot(epochs, train_aucs, label='Training AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.title('Training AUC Curve')
plt.legend(loc='best')
plt.savefig('training_auc_curve.png', dpi=300, bbox_inches='tight')
plt.show()



# 评估模型
final_test_auc = evaluate_model(model, test_loader)
