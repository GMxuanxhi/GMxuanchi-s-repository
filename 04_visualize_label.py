import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


font_path = "times.ttf"  # Times New Roman 字体文件路径
tnr = FontProperties(fname=font_path, size=12)

data_dir = 'Distribution_CIFAR_0.3'

# 读取并排序客户端文件
clients = sorted(
    [f for f in os.listdir(data_dir) if f.startswith('train_client_') and f.endswith('.csv')],
    key=lambda x: int(x.split('_')[-1].split('.')[0])
)

# 计算各客户端标签数量
label_counts = {}
for client in clients:
    df = pd.read_csv(os.path.join(data_dir, client))
    counts = df['label'].value_counts().sort_index()
    label_counts[client] = counts

# 构造 DataFrame，行: 客户端, 列: 标签
counts_df = pd.DataFrame(label_counts).T.fillna(0).astype(int)

# 生成简化后的客户端标签列表
client_labels = [f'client{int(name.split("_")[-1].split(".")[0])}' for name in counts_df.index]

# 创建更宽的画布
fig, ax = plt.subplots(figsize=(9, 6))

# 绘制堆叠柱状图（数量）
counts_df.plot(kind='bar', stacked=True, ax=ax)

# 横坐标标签横向显示
ax.set_xticks(range(len(client_labels)))
ax.set_xticklabels(client_labels, fontproperties=tnr, rotation=0, ha='center')

# 设置纵坐标为数量
ax.set_ylabel('Count', fontproperties=tnr)
# ax.set_title('Label Counts per Client', fontproperties=tnr)

# 图例
leg = ax.legend(title='Label', bbox_to_anchor=(1.02, 1), loc='upper left', prop=tnr)
leg.set_title('Label', prop=tnr)

plt.tight_layout()

# 保存并展示
output_path = os.path.join(data_dir, 'label_counts_distribution.png')
plt.savefig(output_path)
plt.show()

print(f"Chart saved to: {output_path}")
