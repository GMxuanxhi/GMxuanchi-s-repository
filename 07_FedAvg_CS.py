import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import os
from sklearn.metrics import roc_auc_score, precision_score, recall_score

last_recon = None

if torch.cuda.is_available():
    print("GPU is available")
else:
    print("GPU is not available")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 创建用于存储输出数据的新文件夹
folder_name = 'FedAvg_CS_0.5_1.5_CIFAR_0.1_1'
print(folder_name)
folder_path = os.path.join(".", folder_name)
os.makedirs(folder_path, exist_ok=True)


# 指定包含测试集和训练集的文件夹路径
data_folder_path = "Distribution_CIFAR_0.1"

# 加载和处理数据集
class ImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.dataframe = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]
        image = Image.open(img_name).convert("RGB")
        label = int(self.dataframe.iloc[idx, 1])
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label


transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),                        
    transforms.Normalize((0.5, 0.5, 0.5),         
                         (0.5, 0.5, 0.5))          
])
                
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)     # 32×32×3 → 32×32×32
        self.pool1 = nn.MaxPool2d(2, 2)                             # 32×32×32 → 16×16×32

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)    # 16×16×32 → 16×16×64
        self.pool2 = nn.MaxPool2d(2, 2)                             # 16×16×64 → 8×8×64

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)    # 8×8×64 → 8×8×64
        self.pool3 = nn.MaxPool2d(2, 2)                             # 8×8×64 → 4×4×64

        self.fc1 = nn.Linear(64 * 4 * 4, 256)                       # 1024 → 256
        self.fc2 = nn.Linear(256, 10)                               # 256 → 10 类别

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))                       # → 16×16×32
        x = self.pool2(F.relu(self.conv2(x)))                       # → 8×8×64
        x = self.pool3(F.relu(self.conv3(x)))                       # → 4×4×64
        x = x.view(x.size(0), -1)                                   # → (batch_size, 1024)
        x = F.relu(self.fc1(x))                                     # → 256
        x = self.fc2(x)                                             # → 10
        return x



def train_model(model, train_data, criterion, optimizer, num_epochs):
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    average_loss = total_loss / len(train_loader)
    return average_loss


def get_sample_counts(data_folder_path, i_range):
    sample_counts = []
    for i in i_range:
        csv_file = os.path.join(data_folder_path, f'train_client_{i+1}.csv')
        df = pd.read_csv(csv_file)
        sample_counts.append(len(df))
    return sample_counts

def prune_model(state_dict, sparsity):
    all_weights = torch.cat([param.flatten() for param in state_dict.values()])
    k = int(len(all_weights) * sparsity)
    threshold = all_weights.abs().kthvalue(k).values.item()

    pruned_state_dict = {}
    mask = {}

    for key, param in state_dict.items():
        param_data = param.clone()
        binary_mask = (param_data.abs() > threshold).float()
        pruned_param = param_data * binary_mask
        pruned_state_dict[key] = pruned_param
        mask[key] = binary_mask

    return pruned_state_dict, mask

def complement_sparsify(state_dict, mask):
    # 保留原来为 0 的部分（complement），其余置 0
    sparsified_state = {}
    for key in state_dict.keys():
        sparsified_state[key] = state_dict[key] * (1 - mask[key])
    return sparsified_state

def add_weight_dicts(w1, w2, weight):
    result = {}
    for key in w1:
        result[key] = w1[key] + weight * w2[key]
    return result


def evaluate_model(model, test_data, criterion):
    test_loader = DataLoader(test_data, batch_size= 64, shuffle=False)
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            probs = F.softmax(outputs, dim=1)
            all_probs.extend(probs.detach().cpu().numpy())  # 保存所有类别的概率

    test_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total

    # 多分类 AUC (macro averaged)
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except ValueError:
        auc = None  # 有些特殊情况不能算，比如某一类标签缺失

    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')

    return test_loss, accuracy, auc, precision, recall



def average_weights(state_dicts, sample_counts):
    
    total_samples = sum(sample_counts)
    average_dict = {}
    for key in state_dicts[0].keys():
        weighted_sum = sum(sd[key] * count for sd, count in zip(state_dicts, sample_counts))
        average_dict[key] = weighted_sum / total_samples
    return average_dict


def federated_train_cs(num_rounds, server_sparsity, aggregation_ratio):

    # 初始化模型、客户端列表和损失函数
    global_model = CNN().to(device)
    client_models = [CNN().to(device) for _ in range(10)]
    criterion = nn.CrossEntropyLoss()

    # 将全量（密集）模型作为初始模型
    global_state = global_model.state_dict()
    pruned_global_state = {k: v.clone() for k, v in global_state.items()}
    mask = {k: torch.zeros_like(v) for k, v in global_state.items()}

    # 用于记录每轮评估指标
    accuracy_list, auc_list, precision_list, recall_list, testing_losses = [], [], [], [], []
    

    # 通信轮循环
    for round in range(num_rounds):
        # 聚合前准备客户端更新列表
        client_full_states = []
        client_complement_updates = []

        # 每个客户端本地训练
        for i, model in enumerate(client_models):
            model.load_state_dict(pruned_global_state)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            train_data = ImageDataset(
                csv_file=os.path.join(data_folder_path, f'train_client_{i+1}.csv'),
                transform=transform
            )
            train_model(model, train_data, criterion, optimizer,num_epochs=5)
            full_state = model.state_dict()
            client_full_states.append(full_state)

            # 从第2轮 (round>0) 开始，提取补集稀疏更新
            if round > 0:
                comp_update = complement_sparsify(full_state, mask)
                client_complement_updates.append(comp_update)

        # 服务器端聚合
        if round == 0:
            # 第1轮：标准 FedAvg
            aggregated_model = average_weights(client_full_states, sample_counts)
        else:
            # 后续轮：w' + η' * Σ(补集更新)
            delta = average_weights(client_complement_updates, sample_counts)
            aggregated_model = add_weight_dicts(
                pruned_global_state,
                delta,
                aggregation_ratio
            )

        # 评估：先对完整聚合模型 (dense) 进行评估
        # 使用相同的测试集
        test_data = ImageDataset(
            csv_file=os.path.join(data_folder_path, 'test.csv'),
            transform=transform
        )
        global_model.load_state_dict(aggregated_model)
        dense_loss, dense_acc, dense_auc, dense_prec, dense_rec = evaluate_model(
            global_model, test_data, criterion
        )
        print(
            f"Dense Round {round+1}: Loss={dense_loss:.4f} | "
            f"Acc={dense_acc:.2f}% | Prec={dense_prec:.4f} | Rec={dense_rec:.4f} | "
            f"AUC={dense_auc if dense_auc is not None else 'N/A'}"
        )
        
        accuracy_list.append(dense_acc)
        auc_list.append(dense_auc if dense_auc is not None else 0)
        precision_list.append(dense_prec)
        recall_list.append(dense_rec)
        testing_losses.append(dense_loss)
        

        # 对完整模型进行剪枝，更新 pruned_global_state 和 mask
        pruned_global_state, mask = prune_model(aggregated_model, server_sparsity)
        global_model.load_state_dict(pruned_global_state)

        # # 评估：再对剪枝后的稀疏模型 (sparse) 进行评估
        # test_loss, accuracy, auc, precision, recall = evaluate_model(
        #     global_model, test_data, criterion
        # )
        

        # print(
        #     f"Sparse Round {round+1}: Loss={test_loss:.4f} | "
        #     f"Acc={accuracy:.2f}% | Prec={precision:.4f} | Rec={recall:.4f} | "
        #     f"AUC={auc if auc is not None else 'N/A'}"
        # )

    return accuracy_list, auc_list, precision_list, recall_list, testing_losses





# 弄清楚数量
sample_counts = get_sample_counts(data_folder_path,[0,1,2,3,4,5,6,7,8,9])


# 训练并评估联邦学习模型
accuracy_list, auc_list, precision_list, recall_list, testing_losses = federated_train_cs(num_rounds=50, server_sparsity=0.5, aggregation_ratio=1.5)

# 保存结果
accuracy_df = pd.DataFrame(accuracy_list, columns=["Accuracy"])
accuracy_df.to_csv(os.path.join(folder_path, 'accuracy.csv'), index=False)

auc_df = pd.DataFrame(auc_list, columns=["AUC"])
auc_df.to_csv(os.path.join(folder_path, 'auc.csv'), index=False)

precision_df = pd.DataFrame(precision_list, columns=["Precision"])
precision_df.to_csv(os.path.join(folder_path, 'precision.csv'), index=False)

recall_df = pd.DataFrame(recall_list, columns=["Recall"])
recall_df.to_csv(os.path.join(folder_path, 'recall.csv'), index=False)

test_df = pd.DataFrame(testing_losses, columns=["Testing Loss"])
test_df.to_csv(os.path.join(folder_path, 'testing_loss.csv'), index=False)

# 绘制准确率折线图
plt.figure(figsize=(10, 6))  # 设置图像大小
plt.plot(range(50), accuracy_list)
plt.xlim(0, 50)
plt.ylim(0, 100)
plt.xlabel('Rounds')
plt.ylabel('Test Accuracy')
plt.savefig(os.path.join(folder_path, 'accuracy_plot.png'))
plt.close()  # 关闭当前的图形，避免绘图冲突