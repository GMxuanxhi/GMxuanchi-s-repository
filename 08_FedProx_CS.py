import pandas as pd 
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from sklearn.metrics import roc_auc_score, precision_score, recall_score

# 超参数项
MU = 0.01
SP = 0.2
AR = 1.5
AL = 1

if torch.cuda.is_available():
    print("GPU is available")
else:
    print("GPU is not available")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 创建用于存储输出数据的新文件夹
folder_name = f'FedProx_CS_{SP}_{AR}_{MU}_CIFAR_{AL}_1'
folder_path = os.path.join(".", folder_name)
os.makedirs(folder_path, exist_ok=True)

# 指定包含测试集和训练集的文件夹路径
data_folder_path = f"Distribution_CIFAR_{AL}"
print(data_folder_path)

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


def train_model(model, train_data, criterion, optimizer, num_epochs, mu=0.0, global_state=None):
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            # FedProx proximal 正则项
            if mu > 0.0 and global_state is not None:
                prox_reg = 0.0
                for name, param in model.named_parameters():
                    prox_reg += torch.sum((param - global_state[name].to(device))**2)
                loss = loss + (mu / 2) * prox_reg
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
        pruned_state_dict[key] = param_data * binary_mask
        mask[key] = binary_mask
    return pruned_state_dict, mask


def complement_sparsify(state_dict, mask):
    return {key: state_dict[key] * (1 - mask[key]) for key in state_dict.keys()}

def add_weight_dicts(w1, w2, weight):
    return {key: w1[key] + weight * w2[key] for key in w1}


def evaluate_model(model, test_data, criterion):
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    model.eval()
    correct = total = 0
    total_loss = 0
    all_labels, all_preds, all_probs = [], [], []

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
            all_probs.extend(F.softmax(outputs, dim=1).cpu().numpy())

    test_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except ValueError:
        auc = None
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


def federated_train_cs(num_rounds, server_sparsity, aggregation_ratio, mu):
    global_model = CNN().to(device)
    client_models = [CNN().to(device) for _ in range(10)]
    criterion = nn.CrossEntropyLoss()
    global_state = global_model.state_dict()
    pruned_global_state = {k: v.clone() for k, v in global_state.items()}
    mask = {k: torch.zeros_like(v) for k, v in global_state.items()}

    accuracy_list, auc_list, precision_list, recall_list, testing_losses = [], [], [], [], []
    sample_counts = get_sample_counts(data_folder_path, range(10))

    for round in range(num_rounds):
        client_full_states = []
        client_complement_updates = []
        for i, model in enumerate(client_models):
            model.load_state_dict(pruned_global_state)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            train_data = ImageDataset(
                csv_file=os.path.join(data_folder_path, f'train_client_{i+1}.csv'),
                transform=transform
            )
            # 使用 FedProx 本地训练
            train_model(model, train_data, criterion, optimizer, num_epochs=5, mu=mu, global_state=pruned_global_state)
            full_state = model.state_dict()
            client_full_states.append(full_state)
            if round > 0:
                comp_update = complement_sparsify(full_state, mask)
                client_complement_updates.append(comp_update)

        if round == 0:
            aggregated_model = average_weights(client_full_states, sample_counts)
        else:
            delta = average_weights(client_complement_updates, sample_counts)
            aggregated_model = add_weight_dicts(
                pruned_global_state,
                delta,
                aggregation_ratio
            )

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

        pruned_global_state, mask = prune_model(aggregated_model, server_sparsity)
        global_model.load_state_dict(pruned_global_state)

    return accuracy_list, auc_list, precision_list, recall_list, testing_losses


sample_counts = get_sample_counts(data_folder_path,[0,1,2,3,4,5,6,7,8,9])
accuracy_list, auc_list, precision_list, recall_list, testing_losses = federated_train_cs(num_rounds=50, server_sparsity=SP, aggregation_ratio=AR, mu=MU)

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
plt.figure(figsize=(10, 6))  
plt.plot(range(50), accuracy_list)
plt.xlim(0, 50)
plt.ylim(0, 100)
plt.xlabel('Rounds')
plt.ylabel('Test Accuracy')
plt.savefig(os.path.join(folder_path, 'accuracy_plot.png'))
plt.close()
