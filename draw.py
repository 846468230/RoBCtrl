import networkx as nx
import matplotlib.pyplot as plt
import torch_geometric
from torch_geometric.utils import to_networkx
from dataset import SocialBotDataset
data = SocialBotDataset(root="./data",dataset="twibot-20")
# 将pyg的数据转换为NetworkX图形对象
graph = to_networkx(data[0])
source = 3841 #[4688,3841,341,6478,5941]
# 获取2阶邻居
two_hop_neighbors = nx.single_source_shortest_path_length(graph, source, cutoff=2)

# 添加标签
labels = {}
for node in two_hop_neighbors:
    if node in data[0].y:
        labels[node]= data[0].y[node]
    else:
        labels[node] = 2

# 绘制图形
pos = nx.spring_layout(graph)
nx.draw(graph, pos, node_color=[labels[node] for node in graph.nodes()], with_labels=True)
plt.show()
