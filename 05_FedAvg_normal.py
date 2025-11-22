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

last_recon = None

if torch.cuda.is_available():
    print("GPU is available")

else:
    print("GPU is not available")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 创建用于存储输出数据的新文件夹
folder_name = 'FedAvg_CIFAR_0.1_1'
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
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)     
        self.pool1 = nn.MaxPool2d(2, 2)                             

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)    
        self.pool2 = nn.MaxPool2d(2, 2)                             

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)    
        self.pool3 = nn.MaxPool2d(2, 2)                            

        self.fc1 = nn.Linear(64 * 4 * 4, 256)               
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))          
        x = self.pool2(F.relu(self.conv2(x)))              
        x = self.pool3(F.relu(self.conv3(x)))       
        x = x.view(x.size(0), -1)                                   
        x = F.relu(self.fc1(x))                            
        x = self.fc2(x)                         
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



def average_weights(state_dicts, sample_counts):
    average_dict = {}
    total_samples = sum(sample_counts)
    for key in state_dicts[0].keys():
        key_params = []
        for state_dict, count in zip(state_dicts, sample_counts):
            key_params.append(state_dict[key] * count)
        weighted_sum = sum(key_params)

        # 计算加权平均值，并存入average_dict
        average_dict[key] = weighted_sum / total_samples

    # 返回计算得到的加权平均权重字典
    return average_dict

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
            all_probs.extend(probs.detach().cpu().numpy())

    test_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total

    # 多分类 AUC (macro averaged)
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except ValueError:
        auc = None 

    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')

    return test_loss, accuracy, auc, precision, recall



def federated_train(num_rounds): 
    global_model = CNN().to(device) 
    client_models = [CNN().to(device) for _ in range(10)] 
    criterion = nn.CrossEntropyLoss()
    accuracy_list   = []
    auc_list        = []
    precision_list  = []
    recall_list     = []
    testing_losses  = []

    for round in range(num_rounds):
        global_state_dict = global_model.state_dict()
        
        # 为每个客户端模型加载特定的数据集并进行训练
        for client_index, model in enumerate(client_models):
            model.load_state_dict(global_state_dict)
            optimizer = optim.Adam(model.parameters(), lr = 0.001)
            train_data = ImageDataset(csv_file=os.path.join( data_folder_path, f'train_client_{client_index+1}.csv'), transform=transform)
            train_model(model, train_data, criterion, optimizer,num_epochs=5)

        
        # 在所有客户端模型上平均权重
        global_state_dict = average_weights([model.state_dict() for model in client_models], sample_counts = sample_counts)
        global_model.load_state_dict(global_state_dict)
    
        # 使用更新后的全局模型评估测试集性能
        test_data = ImageDataset(csv_file=os.path.join(data_folder_path, 'test.csv'), transform=transform)
        test_loss, accuracy, auc, precision, recall = evaluate_model(global_model, test_data, criterion)
        testing_losses.append(test_loss)
        accuracy_list.append(accuracy)
        auc_list.append(auc if auc is not None else 0 )
        precision_list.append(precision)
        recall_list.append(recall)
        print(f"\nRound {round+1}\nTest Loss: {test_loss:.4f} | Accuracy: {accuracy:.2f}% | Precision: {precision:.4f} | Recall: {recall:.4f} | AUC: {auc:.4f}" if auc is not None else f"\nRound {round+1}\nTest Loss: {test_loss:.4f} | Accuracy: {accuracy:.2f}% | Precision: {precision:.4f} | Recall: {recall:.4f} | AUC: N/A")

    return accuracy_list, auc_list, precision_list, recall_list, testing_losses



# 弄清楚数量
sample_counts = get_sample_counts(data_folder_path,[0,1,2,3,4,5,6,7,8,9])


# 训练并评估联邦学习模型
accuracy_list, auc_list, precision_list, recall_list, testing_losses = federated_train(num_rounds=50)

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