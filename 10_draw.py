import os
import pandas as pd
import matplotlib.pyplot as plt

# 设置结果文件夹的路径
result_folders = [
    # 'result_FedAvg_C',
    # 'result_FedProx0.001_CIFAR_3',
    # 'result_FedProx0.005_CIFAR_3',
    # 'result_FedProx0.01_CIFAR_3',
    # 'result_FedProx0.1_CIFAR_3',
    # 'result_FedAvg_CS_0.5_5_CIFAR_3',
    # 'result_FedProx0.005_CIFAR_0.3',
    # 'result_FedAvg_CS_0.5_10_CIFAR_3',
    # 'result_FedProx0.005_CIFAR_1',
    'result_FedAvg_CIFAR_0.1',
    'result_FedAvg_CIFAR_0.3',
    'result_FedAvg_CIFAR_1',
    'result_FedAvg_CIFAR_3',
    'result_FedAvg_CS_0.5_5_CIFAR_3',
    # 'result_FedProx0.005_CIFAR_3',
]

# 设置保存路径
save_dir = 'result'
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'FedAvg_CS_accuracy_comparison.png')

plt.figure(figsize=(10, 6))

# 遍历每个文件夹
for folder in result_folders:
    csv_path = os.path.join(folder, 'accuracy.csv')
    if os.path.exists(csv_path):
        # 读取CSV文件并去掉标题行
        data = pd.read_csv(csv_path)
        # 绘图：横轴是索引（从0开始），纵轴是准确率
        plt.plot(data.index, data['Accuracy'], label=folder)

# 图形设置
# plt.title('Federated Learning Accuracy Comparison')
plt.xlabel('Round')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

# **这里是修改的部分**
plt.xlim(0, len(data) - 1)  # 横轴从 0 到 最大索引

# 保存图片到result文件夹
plt.savefig(save_path, dpi=300)
plt.show()

print(f"图片已保存到: {save_path}")
