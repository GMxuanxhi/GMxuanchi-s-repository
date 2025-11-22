# 一共要改三个地方
import pandas as pd
import numpy as np
import os
import shutil

# 加载带有两列（path, label）的数据
def load_data(file_path='CIFAR10Train.csv'):
    return pd.read_csv(file_path)

# 生成 Dirichlet 分布矩阵：行数 = 标签数，列数 = 客户端数
def generate_dirichlet(alpha, num_clients, num_labels):
    return np.random.dirichlet([alpha] * num_clients, size=num_labels)

# 按标签和 Dirichlet 分布划分数据
def split_dirichlet(data, distributions, num_clients):
    clients = {i: [] for i in range(num_clients)} 
    labels = sorted(data['label'].unique())
    for dist_row, label in zip(distributions, labels):
        subset = data[data['label'] == label]
        n = len(subset)
        counts = np.round(dist_row * n).astype(int)
        counts[-1] = n - counts[:-1].sum()
        perm = np.random.permutation(n)
        start = 0
        for client_idx, cnt in enumerate(counts):
            if cnt > 0:
                clients[client_idx].append(subset.iloc[perm[start:start+cnt]])
            start += cnt

    for i in clients:
        if clients[i]:
            clients[i] = pd.concat(clients[i], ignore_index=True)
        else:
            clients[i] = pd.DataFrame(columns=data.columns)
    return clients

# 保存每个客户端的数据到独立 CSV
def save_clients(clients, base_dir):
    os.makedirs(base_dir, exist_ok=True)
    for i, df in clients.items():
        df.to_csv(os.path.join(base_dir, f'train_client_{i+1}.csv'), index=False)

# 拷贝并重命名 testAll.csv
def copy_test(src='CIFAR10Test.csv', dest_dir='Distribution_CIFAR_3'):
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copy(src, os.path.join(dest_dir, 'test.csv'))
    print(f"Copied {src} → {dest_dir}/test.csv")

def main():
    data = load_data('CIFAR10Train.csv')
    num_clients = 10
    labels = sorted(data['label'].unique())
    distributions = generate_dirichlet(alpha=3, num_clients=num_clients, num_labels=len(labels))

    # —— 计算方差并求和 —— #
    variances = np.var(distributions, axis=1)
    sum_variances = float(np.sum(variances))
    print("Dirichlet 分布：\n", distributions)
    print("各行方差：", variances)
    print("方差之和：", sum_variances)

    # —— 指定输出目录，并保存 TXT 到该目录 —— #
    output_dir = 'Distribution_CIFAR_3'
    os.makedirs(output_dir, exist_ok=True)
    txt_path = os.path.join(output_dir, 'distribution_stats.txt')
    with open(txt_path, 'w') as f:
        f.write("Dirichlet 分布矩阵（每行→一个标签，每列→一个客户端）：\n")
        f.write(np.array2string(distributions, precision=4, separator=', '))
        f.write("\n\n各行方差：\n")
        f.write(np.array2string(variances, precision=6, separator=', '))
        f.write(f"\n\n方差之和：{sum_variances}\n")
    print(f"Saved distribution stats to {txt_path}")

    # 划分并保存客户端数据
    clients = split_dirichlet(data, distributions, num_clients)
    save_clients(clients, base_dir=output_dir)
    copy_test(dest_dir=output_dir)

if __name__ == "__main__":
    main()
