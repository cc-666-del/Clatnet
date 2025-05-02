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

# 数据预处理：将所有特征展平并拼接为一个输入
def flatten_features(seq_data, tfidf_data, static_data, time_data):
    # 展平序列特征
    seq_data_flat = seq_data.view(seq_data.size(0), -1)
    # 展平时间特征
    time_data_flat = time_data.view(time_data.size(0), -1)
    # 拼接所有特征
    combined_features = torch.cat([seq_data_flat, tfidf_data, static_data, time_data_flat], dim=1)
    return combined_features

# 准备训练和测试数据
X_train = flatten_features(X_train_seq, X_train_tfidf, X_train_static, X_train_time)
X_test = flatten_features(X_test_seq, X_test_tfidf, X_test_static, X_test_time)

# 创建 DataLoader
batch_size = 64
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义 MLP 模型
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], output_dim=1, dropout_prob=0.4):
        super(MLP, self).__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# 获取输入特征的维度
input_dim = X_train.shape[1]

# 初始化模型
model = MLP(input_dim=input_dim, hidden_dims=[256, 128, 64], output_dim=1, dropout_prob=0.4)

# 将模型移至CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
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
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            predicted = (probs > 0.5).astype(float)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted)
            all_probs.extend(probs)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs).flatten()

    # 保存真实标签和预测概率
    np.save('mlp_y_test2.npy', all_labels)
    np.save('mlp_y_pred_proba2.npy', all_probs)

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
    plt.savefig('pr_curve_mlp2.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 绘制并保存混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.title('Confusion Matrix - MLP Model')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix_mlp2.png', dpi=300, bbox_inches='tight')
    plt.show()

# 开始训练和评估
train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=10)
evaluate_model(model, test_loader)
