import os
import pandas as pd

# 指定数据集根目录
root_dir = "dataset/CIFAR10/train"

# 遍历子文件夹并分配标签
class_names = sorted(os.listdir(root_dir))  # 确保顺序一致
class_to_label = {name: idx for idx, name in enumerate(class_names)}

# 收集所有图片路径和标签
data = []
for class_name in class_names:
    class_folder = os.path.join(root_dir, class_name)
    if os.path.isdir(class_folder):
        for file in os.listdir(class_folder):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(class_folder, file)
                label = class_to_label[class_name]
                data.append([file_path, label])

# 保存为CSV
df = pd.DataFrame(data, columns=['image_path', 'label'])
df.to_csv('CIFAR10Train.csv', index=False)

print("CSV 文件已生成，包含 {} 个图像条目。".format(len(df)))

# 指定数据集根目录
root_dir = "dataset/CIFAR10/test"

# 遍历子文件夹并分配标签
class_names = sorted(os.listdir(root_dir))  # 确保顺序一致
class_to_label = {name: idx for idx, name in enumerate(class_names)}

# 收集所有图片路径和标签
data = []
for class_name in class_names:
    class_folder = os.path.join(root_dir, class_name)
    if os.path.isdir(class_folder):
        for file in os.listdir(class_folder):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(class_folder, file)
                label = class_to_label[class_name]
                data.append([file_path, label])

# 保存为CSV
df = pd.DataFrame(data, columns=['image_path', 'label'])
df.to_csv('CIFAR10Test.csv', index=False)

print("CSV 文件已生成，包含 {} 个图像条目。".format(len(df)))