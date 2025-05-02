# 导入必要的库
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (roc_auc_score, precision_recall_curve, auc,
                             classification_report, confusion_matrix)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import math

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


# 更新 ICD 编码映射，添加 UNK 令牌
icd_mapping = {'<UNK>': 0}
icd_list = list(set(X_train_seq[:, :, 0].flatten().tolist() +
                    X_test_seq[:, :, 0].flatten().tolist()))
for idx, icd in enumerate(icd_list):
    if icd not in icd_mapping:
        icd_mapping[icd] = len(icd_mapping)
idx_to_icd = {idx: icd for icd, idx in icd_mapping.items()}

# 将 ICD 编码映射到索引
def map_icd_to_ids(seq_data, mapping):
    mapped_seq = seq_data.clone()
    for i in range(seq_data.size(0)):
        for j in range(seq_data.size(1)):
            icd_code = seq_data[i, j, 0].item()
            mapped_seq[i, j, 0] = mapping.get(icd_code, 0)
    return mapped_seq

X_train_seq = map_icd_to_ids(X_train_seq, icd_mapping)
X_test_seq = map_icd_to_ids(X_test_seq, icd_mapping)

# 准备 Transformer 输入
def prepare_transformer_input(seq_data):
    features = seq_data[:, :, 0].long()  # ICD 编码索引
    time_steps = seq_data[:, :, 1]  # 相对天数
    return features, time_steps

X_train_features, X_train_time_steps = prepare_transformer_input(X_train_seq)
X_test_features, X_test_time_steps = prepare_transformer_input(X_test_seq)

# 创建 DataLoader
batch_size = 64
train_dataset = TensorDataset(X_train_features, X_train_time_steps,
                              X_train_tfidf, X_train_static, y_train)
test_dataset = TensorDataset(X_test_features, X_test_time_steps,
                             X_test_tfidf, X_test_static, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size,
                         shuffle=False)

# 定义位置编码器
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len,
                                dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置
        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# 定义 Transformer 模型
class TransformerModel(nn.Module):
    def __init__(self, num_tokens, d_model, nhead, num_layers,
                 dim_feedforward, max_seq_length, input_dim_tfidf,
                 input_dim_static, output_dim=1, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(num_tokens, d_model,
                                      padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, dropout,
                                              max_len=max_seq_length)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead,
                                                    dim_feedforward,
                                                    dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers,
                                                         num_layers)
        self.fc_tfidf = nn.Linear(input_dim_tfidf, 128)
        self.relu = nn.ReLU()
        self.fc_static = nn.Linear(input_dim_static, 64)
        self.fc_final = nn.Sequential(
            nn.Linear(d_model + 128 + 64, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )

    def forward(self, src, tfidf_input, static_input):
        src = src.transpose(0, 1)  # [seq_len, batch_size]
        src_key_padding_mask = (src == 0).transpose(0, 1)
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src,
                                          src_key_padding_mask=src_key_padding_mask)
        seq_representation = memory.mean(dim=0)  # [batch_size, d_model]
        tfidf_out = self.relu(self.fc_tfidf(tfidf_input))
        static_out = self.relu(self.fc_static(static_input))
        combined_features = torch.cat([seq_representation, tfidf_out,
                                       static_out], dim=1)
        output = self.fc_final(combined_features)
        return output

# 获取输入特征的维度
num_tokens = len(icd_mapping)
input_dim_tfidf = X_train_tfidf.shape[1]
input_dim_static = X_train_static.shape[1]
max_seq_length = X_train_features.shape[1]

# 初始化模型
model = TransformerModel(
    num_tokens=num_tokens,
    d_model=128,
    nhead=8,
    num_layers=4,
    dim_feedforward=256,
    max_seq_length=max_seq_length,
    input_dim_tfidf=input_dim_tfidf,
    input_dim_static=input_dim_static,
    output_dim=1,
    dropout=0.1
)

# 将模型移至CUDA
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 定义损失函数和优化器
num_positive = (y_train == 1).sum().item()
num_negative = (y_train == 0).sum().item()
pos_weight_value = num_negative / num_positive if num_positive > 0 else 1.0
pos_weight = torch.tensor([pos_weight_value], device=device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 设置学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                 patience=2,
                                                 factor=0.5)

# 训练模型
def train_model(model, train_loader, criterion, optimizer, scheduler,
                num_epochs=20):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for seq_inputs, time_steps, tfidf_inputs, static_inputs, labels in train_loader:
            seq_inputs = seq_inputs.to(device)
            tfidf_inputs = tfidf_inputs.to(device)
            static_inputs = static_inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(seq_inputs, tfidf_inputs, static_inputs)
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
        for seq_inputs, time_steps, tfidf_inputs, static_inputs, labels in test_loader:
            seq_inputs = seq_inputs.to(device)
            tfidf_inputs = tfidf_inputs.to(device)
            static_inputs = static_inputs.to(device)
            labels = labels.to(device)
            outputs = model(seq_inputs, tfidf_inputs, static_inputs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            predicted = (probs > 0.5).astype(float)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted)
            all_probs.extend(probs)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs).flatten()
    np.save('transformer_y_test2.npy', all_labels)
    np.save('transformer_y_pred_proba2.npy', all_probs)
    print("Classification Report:")
    print(classification_report(all_labels, all_preds))
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)
    roc_auc = roc_auc_score(all_labels, all_probs)
    print(f'ROC AUC: {roc_auc:.4f}')
    precision_vals, recall_vals, _ = precision_recall_curve(all_labels,
                                                            all_probs)
    pr_auc = auc(recall_vals, precision_vals)
    print(f'PR AUC: {pr_auc:.4f}')
    plt.figure()
    plt.plot(recall_vals, precision_vals,
             label=f'PR AUC = {pr_auc:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.savefig('pr_curve_transformer2.png', dpi=300,
                bbox_inches='tight')
    plt.show()
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    plt.title('Confusion Matrix - Transformer Model')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix_transformer2.png', dpi=300,
                bbox_inches='tight')
    plt.show()

# 开始训练和评估
train_model(model, train_loader, criterion, optimizer, scheduler,
            num_epochs=20)
evaluate_model(model, test_loader)
