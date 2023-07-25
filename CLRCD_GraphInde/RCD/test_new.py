import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import json
import numpy as np
from sklearn.metrics import roc_auc_score

class GraphDataset(Dataset):
    def __init__(self, user_indices, item_indices, knowledge_indices, labels):
        self.user_indices = user_indices
        self.item_indices = item_indices
        self.knowledge_indices = knowledge_indices
        self.labels = labels
        
    def __len__(self):
        return len(self.user_indices)
    
    def __getitem__(self, index):
        user_index = self.user_indices[index]
        item_index = self.item_indices[index]
        knowledge_index = self.knowledge_indices[index]
        label = self.labels[index]
        
        return user_index, item_index, knowledge_index, label


# 构建数据集
with open('log_data_all.json', 'r') as f:
    data = json.load(f)

user_indices = []
item_indices = []
knowledge_indices = []
labels = []

for item in data:
    user_indices.append(item["user_id"])
    for log in item["logs"]:
        item_indices.append(log["exer_id"])
        knowledge_indices.extend(log["knowledge_code"])
        labels.append(log["result"])  # result represents whether the student answered the question correctly

# 去重索引
user_indices = list(set(user_indices))
item_indices = list(set(item_indices))
knowledge_indices = list(set(knowledge_indices))

# 创建边索引和评分字典（做对为1，做错为-1）
edges = []
interaction_dict = {} # 记录学生与题目的交互情况，形如{(user_id, exer_id): score}
for item in data:
    user_id = item["user_id"]
    for log in item["logs"]:
        exer_id = log["exer_id"]
        score = 2*log["score"] - 1  # 将0,1转化为-1,1
        knowledge_codes = log["knowledge_code"]
        for knowledge_code in knowledge_codes:
            edges.append((user_id, exer_id, knowledge_code))
        interaction_dict[(user_id, exer_id)] = score

# 创建图的邻接矩阵
num_users = len(user_indices)
num_items = len(item_indices)
num_knowledge = len(knowledge_indices)
adj_matrix = torch.zeros((num_users + num_items + num_knowledge, num_users + num_items + num_knowledge))

for user_id, item_id, knowledge_id in edges:
    user_index = user_indices.index(user_id)
    item_index = item_indices.index(item_id)
    knowledge_index = knowledge_indices.index(knowledge_id)
    
    # 设置学生和题目的交互
    interaction_score = interaction_dict.get((user_id, item_id), 0)  # 如果学生没有做过这个题目，分数就是0
    adj_matrix[user_index, item_index + num_users] = interaction_score
    adj_matrix[item_index + num_users, user_index] = interaction_score
    
    # 设置题目和知识点的关联
    adj_matrix[item_index + num_users, knowledge_index + num_users + num_items] = 1
    adj_matrix[knowledge_index + num_users + num_items, item_index + num_users] = 1

# 将PyTorch tensor转换为numpy数组
adj_matrix_np = adj_matrix.numpy()

# 保存到文件
np.save('adj_matrix.npy', adj_matrix_np)

def drop_edges(adj_matrix, drop_ratio=0.1):
    num_edges = torch.sum(adj_matrix)
    num_edges_to_drop = int(drop_ratio * num_edges)
    edge_indices = torch.nonzero(adj_matrix)
    drop_indices = edge_indices[torch.randint(len(edge_indices), (num_edges_to_drop,))]

    adj_matrix_dropped = adj_matrix.clone()
    adj_matrix_dropped[drop_indices[:, 0], drop_indices[:, 1]] = 0
    return adj_matrix_dropped


class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, num_knowledge, embed_dim, num_layers):
        super(LightGCN, self).__init__()
        self.user_embeddings = nn.Embedding(num_users, embed_dim)
        self.item_embeddings = nn.Embedding(num_items, embed_dim)
        self.knowledge_embeddings = nn.Embedding(num_knowledge, embed_dim)
        self.num_layers = num_layers

    def propagate(self, adj_matrix, embeddings):
        for _ in range(self.num_layers):
            adj_matrix = drop_edges(adj_matrix)  # drop edges in each layer
            embeddings = adj_matrix @ embeddings
        return embeddings

    def forward(self, adj_matrix, user_indices, item_indices, knowledge_indices):
        user_emb = self.user_embeddings(user_indices)
        item_emb = self.item_embeddings(item_indices)
        knowledge_emb = self.knowledge_embeddings(knowledge_indices)
        embeddings = torch.cat([user_emb, item_emb, knowledge_emb], dim=0)
        embeddings = self.propagate(adj_matrix, embeddings)
        user_emb, item_emb, knowledge_emb = torch.split(embeddings, [len(user_indices), len(item_indices), len(knowledge_indices)], dim=0)
        return user_emb, item_emb, knowledge_emb


# 损失函数和优化器
# criterion = nn.BCEWithLogitsLoss()
# optimizer = torch.optim.Adam(model.parameters())

# 训练过程
def train(model, dataloader, adj_matrix, criterion, optimizer):
    model.train()
    total_loss = 0
    for user_indices, item_indices, knowledge_indices, labels in dataloader:
        optimizer.zero_grad()
        user_emb, item_emb, knowledge_emb = model(adj_matrix, user_indices, item_indices, knowledge_indices)
        logits = (user_emb * item_emb).sum(1)
        loss = criterion(logits, labels.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

# 测试过程（使用AUC作为评价指标）
def test(model, dataloader, adj_matrix):
    model.eval()
    with torch.no_grad():
        scores = []
        labels = []
        for user_indices, item_indices, knowledge_indices, label in dataloader:
            user_emb, item_emb, knowledge_emb = model(adj_matrix, user_indices, item_indices, knowledge_indices)
            score = (user_emb * item_emb).sum(1)
            scores.append(score)
            labels.append(label)
        scores = torch.cat(scores)
        labels = torch.cat(labels)
        auc = roc_auc_score(labels.cpu().numpy(), scores.cpu().numpy())
    return auc

def main():
    # 加载数据
    user_indices, item_indices, knowledge_indices, edges, adj_matrix, labels = load_data()  # 你需要提供一个这样的函数来加载数据

    # 创建数据集和数据加载器
    dataset = GraphDataset(user_indices, item_indices, knowledge_indices, labels)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 设定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建模型并移到设备上
    num_users = len(user_indices)
    num_items = len(item_indices)
    num_knowledge = len(knowledge_indices)
    model = LightGCN(num_users, num_items, num_knowledge, embed_dim=16)
    model.to(device)

    # 创建损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # 开始训练
    num_epochs = 10  # 可以根据实际需要调整
    for epoch in range(num_epochs):
        train_loss = train(model, dataloader, adj_matrix, criterion, optimizer)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}")

    # 测试模型
    test_auc = test(model, dataloader, adj_matrix)
    print(f"Test AUC: {test_auc}")

if __name__ == "__main__":
    main()
