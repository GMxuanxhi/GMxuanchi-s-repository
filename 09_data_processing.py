# 要改两个值
import os
import pandas as pd

# 设置参数
# folders = [f"FedProx0.005_CIFAR_3_{i}" for i in range(1, 6)]
# metrics = ["accuracy.csv", "auc.csv", "precision.csv", "recall.csv", "testing_loss.csv"]
# result_dir = "result_FedProx0.005_CIFAR_3"
# os.makedirs(result_dir, exist_ok=True)

# folders = [f"FedAvg_CIFAR_0.1_{i}" for i in range(1, 6)]
# metrics = ["accuracy.csv", "auc.csv", "precision.csv", "recall.csv", "testing_loss.csv"]
# result_dir = "result_FedAvg_CIFAR_0.1"
# os.makedirs(result_dir, exist_ok=True)

folders = [f"FedAvg_CS_0.5_5_CIFAR_3_{i}" for i in range(1, 6)]
metrics = ["accuracy.csv", "auc.csv", "precision.csv", "recall.csv", "testing_loss.csv"]
result_dir = "result_FedAvg_CS_0.5_5_CIFAR_3"
os.makedirs(result_dir, exist_ok=True)

# 处理每种指标
for metric in metrics:
    df_sum = None
    count = 0

    for folder in folders:
        file_path = os.path.join(folder, metric)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, header=0)  # 正确读取表头
            df = df.apply(pd.to_numeric, errors='coerce')  # 转为数值，非数值变 NaN
            if df_sum is None:
                df_sum = df
            else:
                df_sum = df_sum.add(df, fill_value=0)
            count += 1
        else:
            print(f"缺失文件: {file_path}")

    if df_sum is not None and count > 0:
        df_avg = df_sum / count
        output_path = os.path.join(result_dir, metric)
        df_avg.to_csv(output_path, index=False)
        print(f"保存平均文件: {output_path}")
    else:
        print(f"无法处理 {metric}，文件缺失或格式错误")
