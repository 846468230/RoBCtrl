import os
from deeprobust.graph.utils import *
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score, average_precision_score
from torch_geometric.utils import to_networkx
import networkx as nx

def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask

def eval_deep(log, loader):

    data_size = len(loader.dataset.indices)
    batch_size = loader.batch_size
    if data_size % batch_size == 0:
        size_list = [batch_size] * (data_size//batch_size)
    else:
        size_list = [batch_size] * \
            (data_size // batch_size) + [data_size % batch_size]

    assert len(log) == len(size_list)

    accuracy, f1_macro, f1_micro, precision, recall = 0, 0, 0, 0, 0

    prob_log, label_log = [], []

    for batch, size in zip(log, size_list):
        pred_y, y = batch[0].data.cpu().numpy().argmax(
            axis=1), batch[1].data.cpu().numpy().tolist()
        prob_log.extend(batch[0].data.cpu().numpy()[:, 1].tolist())
        label_log.extend(y)

        accuracy += accuracy_score(y, pred_y) * size
        f1_macro += f1_score(y, pred_y, average='macro') * size
        f1_micro += f1_score(y, pred_y, average='micro') * size
        precision += precision_score(y, pred_y, zero_division=0) * size
        recall += recall_score(y, pred_y, zero_division=0) * size

    auc = roc_auc_score(label_log, prob_log)
    ap = average_precision_score(label_log, prob_log)

    return accuracy/data_size, f1_macro/data_size, f1_micro/data_size, precision/data_size, recall/data_size, auc, ap


def eval_hin(log):

    pred_y, y = log[0].data.cpu().numpy().argmax(
        axis=1), log[1].data.cpu().numpy().tolist()
    prob_log = log[0].data.cpu().numpy()[:, 1].tolist()
    label_log = y

    accuracy = accuracy_score(y, pred_y)
    f1_macro = f1_score(y, pred_y, average='macro')
    f1_micro = f1_score(y, pred_y, average='micro')
    precision = precision_score(y, pred_y, zero_division=0)
    recall = recall_score(y, pred_y, zero_division=0)

    auc = roc_auc_score(label_log, prob_log)
    ap = average_precision_score(label_log, prob_log)

    return accuracy, f1_macro, f1_micro, precision, recall, auc, ap


def eval_shallow(pred_y, prob_y, y):

    f1_macro = f1_score(y, pred_y, average='macro')
    f1_micro = f1_score(y, pred_y, average='micro')
    accuracy = accuracy_score(y, pred_y)
    precision = precision_score(y, pred_y, zero_division=0)
    recall = recall_score(y, pred_y, zero_division=0)
    auc = roc_auc_score(y, prob_y)
    ap = average_precision_score(y, prob_y)

    return accuracy, f1_macro, f1_micro, precision, recall, auc, ap


def change_dataset_format(dataset, process=False, sparse=True):
    features = dataset.features
    labels = dataset.labels
    adj = dataset.adj
    idx_train = dataset.idx_train
    idx_test = dataset.idx_test
    idx_val = dataset.idx_val
    if process:
        adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False, sparse=sparse)
    return features, adj, labels, idx_train, idx_test, idx_val


def torch_sparse_tensor_to_sparse_mx(torch_sparse):
    """Convert a torch sparse tensor to a scipy sparse matrix."""
    try:
        m_index = torch_sparse._indices().numpy()
    except TypeError:
        m_index = torch_sparse._indices().cpu().numpy()
    row = m_index[0]
    col = m_index[1]
    try:
        data = torch_sparse._values().numpy()
    except TypeError:
        data = torch_sparse._values().cpu().numpy()

    sp_matrix = sp.csr_matrix((data, (row, col)), shape=(torch_sparse.size()[0], torch_sparse.size()[1]))

    return sp_matrix



def get_device(num_device=None, cpu=False):
    import torch
    if cpu or num_device==-1:
        return torch.device('cpu')
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{num_device}'
        import torch.cuda
        return torch.device('cuda',num_device)
    else:
        return torch.device('cpu')
    return torch.device('cpu')


def preprocess(adj, features, labels, preprocess_adj=False, preprocess_feature=False, sparse=False, device='cpu'):
    """Convert adj, features, labels from array or sparse matrix to
    torch Tensor, and normalize the input data.
    Parameters
    ----------
    adj : scipy.sparse.csr_matrix
        the adjacency matrix.
    features : scipy.sparse.csr_matrix
        node features
    labels : numpy.array
        node labels
    preprocess_adj : bool
        whether to normalize the adjacency matrix
    preprocess_feature : bool
        whether to normalize the feature matrix
    sparse : bool
       whether to return sparse tensor
    device : str
        'cpu' or 'cuda'
    """

    if preprocess_adj:
        adj = normalize_adj(adj)

    if preprocess_feature:
        features = normalize_feature(features)

    labels = torch.LongTensor(labels)
    if sparse:
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        features = sparse_mx_to_torch_sparse_tensor(features)
    else:
        if sp.issparse(features):
            features = torch.FloatTensor(np.array(features.todense()))
        else:
            features = torch.FloatTensor(features)
        adj = torch.FloatTensor(adj.todense())
    return adj.to(device), features.to(device), labels.to(device)


def add_nodes(self, features, adj, labels, idx_train, target_node, n_added=1, n_perturbations=10):
    print('number of pertubations: %s' % n_perturbations)
    N = adj.shape[0]
    D = features.shape[1]
    modified_adj = reshape_mx(adj, shape=(N + n_added, N + n_added))
    modified_features = self.reshape_mx(features, shape=(N + n_added, D))

    diff_labels = [l for l in range(labels.max() + 1) if l != labels[target_node]]
    diff_labels = np.random.permutation(diff_labels)
    possible_nodes = [x for x in idx_train if labels[x] == diff_labels[0]]

    return modified_adj, modified_features


def generate_injected_features(features, n_added):
    # TODO not sure how to generate features of injected nodes
    features = features.tolil()
    avg = np.tile(features.mean(0), (n_added, 1))
    features[-n_added:] = avg + np.random.normal(0, 1, (n_added, features.shape[1]))
    return features


def injecting_nodes(data, args):
    '''
        injecting nodes to adj, features, and assign labels to the injected nodes
    '''
    adj, features, labels = data.adj, data.features, data.labels
    # features = normalize_feature(features)
    N = adj.shape[0]
    D = features.shape[1]

    n_added = 50 # int(args.ratio * N)
    print('number of injected nodes: %s' % n_added)

    data.adj = reshape_mx(adj, shape=(N + n_added, N + n_added))
    enlarged_features = reshape_mx(features, shape=(N + n_added, D))
    data.features = generate_injected_features(enlarged_features, n_added)
    data.features = normalize_feature(data.features)

    injected_labels = np.random.choice(labels.max() + 1, n_added)
    data.labels = np.hstack((labels, injected_labels))


# def select_nodes(target_gcn, idx_test, labels):
#     '''
#     selecting nodes as reported in nettack paper:
#     (i) the 10 nodes with highest margin of classification, i.e. they are clearly correctly classified,
#     (ii) the 10 nodes with lowest margin (but still correctly classified) and
#     (iii) 20 more nodes randomly
#     '''
#     target_gcn.eval()
#     output = target_gcn.predict()
#
#     margin_dict = {}
#     for idx in idx_test:
#         margin = classification_margin(output[idx], labels[idx])
#         if margin < 0:  # only keep the nodes correctly classified
#             continue
#         margin_dict[idx] = margin
#     sorted_margins = sorted(margin_dict.items(), key=lambda x: x[1], reverse=True)
#     high = [x for x, y in sorted_margins[: 10]]
#     low = [x for x, y in sorted_margins[-10:]]
#     other = [x for x, y in sorted_margins[10: -10]]
#     other = np.random.choice(other, 20, replace=False).tolist()
#
#     return high + low + other

def select_nodes(target_gcn, idx_test, labels):
    node_list = []
    for idx in idx_test :
        if labels[idx] ==1:
            node_list.append(idx)
    # '''
    # selecting nodes as reported in nettack paper:
    # (i) the 10 nodes with highest margin of classification, i.e. they are clearly correctly classified,
    # (ii) the 10 nodes with lowest margin (but still correctly classified) and
    # (iii) 20 more nodes randomly
    # '''
    # target_gcn.eval()
    # output = target_gcn.predict()
    #
    # margin_dict = {}
    # for idx in idx_test:
    #     margin = classification_margin(output[idx], labels[idx])
    #     if margin < 0:  # only keep the nodes correctly classified
    #         continue
    #     margin_dict[idx] = margin
    # sorted_margins = sorted(margin_dict.items(), key=lambda x: x[1], reverse=True)
    # high = [x for x, y in sorted_margins[: 10]]
    # low = [x for x, y in sorted_margins[-10:]]
    # other = [x for x, y in sorted_margins[10: -10]]
    # other = np.random.choice(other, 20, replace=False).tolist()
    #
    # return high + low + other
    return node_list


def single_test(adj, features, target_node, labels, gcn=None):
    output = gcn.predict(features, adj)
    probs = torch.exp(output[[target_node]])

    # acc_test = accuracy(output[[target_node]], labels[target_node])
    acc_test = (output.argmax(1)[target_node] == labels[target_node])
    return acc_test.item()

def draw_heat_map(dataset,append_edges,generated_indice,test_bots_indices):
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib as mpl
    import seaborn as sns
    import pandas as pd
    fontsize = 20
    plt.rc('font', family='Times New Roman', size=fontsize)
    # sns.set(font_scale=1.5)
    st_botb = 0
    st_humanh = 0
    st_bhuman = 0
    st_hbot = 0
    graph = to_networkx(dataset[0])

    all_st_botb = 0
    all_st_humanh = 0
    all_st_bhuman = 0
    all_st_hbot = 0

    for test_indice in test_bots_indices:
        edges = graph.adj[test_indice]
        for key,item in edges.items():
            source = key
            target = test_indice
            if source>len(dataset[0].y):
                continue
            if dataset[0].y[source]==1:
                if dataset[0].y[target] == 1:
                    all_st_botb += 1
                else:
                    all_st_bhuman += 1
            else:
                if dataset[0].y[target]==1:
                    all_st_hbot +=1
                else:
                    all_st_humanh+=1
    all_edges = all_st_botb + all_st_humanh + all_st_bhuman + all_st_hbot
    distribute_orign = [all_st_botb/all_edges,all_st_humanh/all_edges,all_st_bhuman/all_edges,all_st_hbot/all_edges]
    origin_data = {"human":{"human":distribute_orign[1],"bot":distribute_orign[3]},"bot":{"human":distribute_orign[2],"bot":distribute_orign[0]}}
    origin_data = pd.DataFrame(origin_data)
    for edges in append_edges:
        for i in range(len(edges[0])):
            source = edges[0][i] if edges[0][i] < dataset[0].num_nodes  else  generated_indice[edges[0][i] - dataset[0].num_nodes]
            target = edges[1][i] if edges[1][i] < dataset[0].num_nodes  else  generated_indice[edges[1][i] - dataset[0].num_nodes]
            if dataset[0].y[source] == 1:
                if dataset[0].y[target] == 1:
                    st_botb+=1
                else:
                    st_bhuman+=1
            else:
                if dataset[0].y[target] == 1:
                    st_hbot+=1
                else:
                    st_humanh+=1
    fmt = '.2f'
    all_edges_addded = st_botb + st_humanh + st_bhuman + st_hbot + all_edges
    distribute_addededges = [(st_botb+all_st_botb) / all_edges_addded, (st_humanh+ all_st_humanh )/ all_edges_addded, (st_bhuman+all_st_bhuman)/ all_edges_addded, (st_hbot+all_st_hbot) / all_edges_addded]
    ax = sns.heatmap(data=origin_data,cmap=sns.color_palette("rocket", as_cmap=True),fmt=fmt,annot=True,vmin=0.1,vmax=0.35,annot_kws={"fontsize":fontsize,"family":"Times New Roman"},cbar_kws={"orientation":"horizontal"},linewidths=0.05,square=True)#,linewidth=0.05,linecolor="red")
    ax.xaxis.tick_top()
    ax.yaxis.set_label_position("right")
    plt.xticks(fontsize=fontsize,fontfamily="Times New Roman")
    plt.yticks(fontsize=fontsize,fontfamily="Times New Roman")
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=fontsize,width=1)
    font = {"family":"Times New Roman","weight":"normal","size":fontsize}
    plt.xlabel("source",font)
    plt.ylabel("target", font)
    plt.savefig(f'./{dataset.cur_dataset}_origin_heatmap.pdf')
    plt.show()
    # ax.set_yticaklabels()
    # ax.set_xticklabels(ax.get_xticklabels())
    added_data = {"human":{"human":distribute_addededges[1],"bot":distribute_addededges[3]},"bot":{"human":distribute_addededges[2],"bot":distribute_addededges[0]}}
    added_data = pd.DataFrame(added_data)
    ax = sns.heatmap(data=added_data,cmap=sns.color_palette("rocket", as_cmap=True),annot=True,fmt=fmt,vmin=0.1,vmax=0.35,annot_kws={"fontsize":fontsize,"family":"Times New Roman"},cbar_kws={"orientation":"horizontal"},linewidths=0.05,square=True)#,linewidth=0.05,linecolor="red")
    ax.xaxis.tick_top()
    ax.yaxis.set_label_position("right")
    plt.xticks(fontsize=fontsize,fontfamily="Times New Roman")
    plt.yticks(fontsize=fontsize,fontfamily="Times New Roman")
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=fontsize,width=1)
    font = {"family":"Times New Roman","weight":"normal","size":fontsize}
    plt.xlabel("source",font)
    plt.ylabel("target", font)
    plt.savefig(f'./{dataset.cur_dataset}_added_heatmap.pdf')
    plt.show()

    print()

def draw_edge_distribution(dataset,append_edges,generated_indice,test_bots_indices,train_indices):
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib as mpl
    import seaborn as sns
    import pandas as pd
    fontsize = 25
    # sns.set(font_scale=1.5)
    st_botb = 0
    st_humanh = 0
    st_bhuman = 0
    st_hbot = 0
    ori_graph = to_networkx(dataset[0])
    ori_distribution = {}
    new_distribution = {}
    generated_indice = [number.item()  for number in generated_indice]
    train_indices = [number.item() for number in train_indices]
    for indice in test_bots_indices+train_indices:
        ori_distribution[indice] = ori_graph.degree[indice]
    for indice in generated_indice:
        ori_distribution[indice] = 0
    for edges in append_edges:
        for i in range(len(edges[0])):
            ori_graph.add_edge(edges[0][i],edges[1][i])
    for indice in test_bots_indices+train_indices+generated_indice:
        new_distribution[indice] = ori_graph.degree[indice]
    ori_count_distribution = [0] * 6000
    for key, item in ori_distribution.items():
        ori_count_distribution[item] += 1
    new_count_distribution = [0] * 6000
    for key, item in new_distribution.items():
        new_count_distribution[item] += 1
    import matplotlib.pyplot as plt
    index = 2000
    x = list(range(index))
    y1 = ori_count_distribution[:index]
    y2 = new_count_distribution[:index]
    import seaborn as sns
    import pandas as pd
    fontsize = 25
    plt.rc('font', family='Times New Roman', size=fontsize)
    # origin_data = {"degrees":{'origin':y1,'attacked':y2}}
    # origin_data = pd.DataFrame(origin_data)
    fig, ax = plt.subplots(figsize = (6.5,8))

    palette = sns.color_palette("rocket_r")
    data = pd.DataFrame(data=[ [item[0],item[1]]for item in zip(y1,y2)],
                 columns=['Origin', 'Attacked'],
                 index=x)
    # ax.plot(data=data["Attacked"],color='#153aab',label='Attacked')
    # ax.plot(data=data["Origin"],color="#fdcf41",linestyle='-', linewidth=1, label='Origin')
    # plt.legend(loc='upper center', frameon=True, fontsize=fontsize)

    plt.plot(x, y1,color="black",label='Original-Degrees',linewidth=7)
    plt.plot(x, y2,color="white",label='Attacked-Degrees',linewidth=2)
    font = {"family":"Times New Roman","weight":"normal","size":fontsize}
    if dataset.cur_dataset=="cresci-15":
        plt.ylim(-20, 2000)
        plt.xlim(-1, index)
    elif dataset.cur_dataset=="mgtab":
        plt.ylim(0, 75)
        plt.xlim(0, index)
    plt.xlabel("Degrees",font)
    plt.ylabel("Frequency", font)
    plt.legend(loc='upper center',frameon=True, fontsize=fontsize-3,facecolor='#CFCFCF')
    # plt.margins(x=0,y=0)
    plt.tight_layout()
    plt.savefig(f'./{dataset.cur_dataset}_distribution.pdf')
    plt.show()

    # nx.write_gexf(ori_graph, f"{dataset.cur_dataset}_origin.gexf")
    # nx.draw(ori_graph)
    # plt.show()
    print()