import igraph
import networkx as nx
import numpy as np
import powerlaw
import sys
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from torch_geometric.utils import to_scipy_sparse_matrix
from deeprobust.graph.data import Dataset,Dpr2Pyg, Pyg2Dpr
from defense_methods import GCN_defense
from dataset import index_to_mask
from utils import  torch_sparse_tensor_to_sparse_mx
# import matplotlib.pyplot as plt
# plt.switch_backend('agg')
def cal_attacked(RL_attack,ori_pyg,surrogate,args,edge_index,data):
    # parser.add_argument('--dataset', type=str, default="mgtab", choices=["citeseer","twibot-20","mgtab","mgtab-large","cresci-15"])
    if args.dataset == "mgtab":
        ptb_rates= [0.0035,0.007,0.01]
    elif args.dataset == "cresci-15":
        ptb_rates = [0.15, 0.20, 0.25]
    elif args.dataset == "twibot-20":
        ptb_rates = [0.02, 0.04, 0.06]
    elif args.dataset == "mgtab-large":
        ptb_rates = [0.02, 0.04, 0.06]
    for ptb_rate in ptb_rates:
        args.ptb_rate = ptb_rate
        counts = []
        for item in range(5):
            modified_adj,modified_features = RL_attack(ori_pyg,surrogate,args,edge_index)
            A = to_scipy_sparse_matrix(modified_adj)#.toarray()
            pyg_data = Dpr2Pyg(data)
            pyg_data = pyg_data[0]
            pyg_data.edge_index = modified_adj
            pyg_data.x = modified_features.to(args.device)
            pyg_data.train_mask = index_to_mask(data.idx_train, size=modified_features.size(0))
            pyg_data.val_mask = index_to_mask(data.idx_val, size=modified_features.size(0))
            pyg_data.test_mask = index_to_mask(data.idx_test, size=modified_features.size(0))
            ACC = GCN_defense(pyg_data, args.hid_dim, args.device, epochs=args.epochs, args=args, attacked=True)
            Statistical_data=compute_graph_statistics(A)
            Statistical_data["GCN_ACC"] = ACC
            counts.append(Statistical_data)
        print(f"pertubate_rate : {ptb_rate}")
        for key,item in counts[0].items():
            temp_values =[counts[0][key],counts[1][key],counts[2][key],counts[3][key],counts[4][key]]
            mean = np.mean(temp_values)
            std = np.std(temp_values)
            print(f"{key} : {mean} + {std}", end="\t")
        print()
    sys.exit()




def statistics_degrees(A_in):
    """
    Compute min, max, mean degree

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    d_max. d_min, d_mean
    """

    degrees = A_in.sum(axis=0)
    return np.max(degrees), np.min(degrees), np.mean(degrees)


def statistics_LCC(A_in):
    """
    Compute the size of the largest connected component (LCC)

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Size of LCC

    """

    unique, counts = np.unique(connected_components(A_in)[1], return_counts=True)
    LCC = np.where(connected_components(A_in)[1] == np.argmax(counts))[0]
    return LCC


def statistics_wedge_count(A_in):
    """
    Compute the wedge count of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    The wedge count.
    """
    degrees = A_in.sum(axis=0).flatten()
    return float(np.sum(np.array([0.5 * x * (x - 1) for x in degrees])))


def statistics_claw_count(A_in):
    """
    Compute the claw count of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Claw count
    """

    if sp.issparse(A_in):
        degrees = A_in.sum(axis=1)
    else:
        degrees = A_in.sum(axis=0).flatten()
    return float(np.sum(np.array([1 / 6. * x * (x - 1) * (x - 2) for x in degrees])))


def statistics_triangle_count(A_in):
    """
    Compute the triangle count of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Triangle count
    """
    if sp.issparse(A_in):
        A_graph = nx.from_scipy_sparse_array(A_in)
    else:
        A_graph = nx.from_numpy_matrix(A_in)
    triangles = nx.triangles(A_graph)
    t = np.sum(list(triangles.values())) / 3
    return int(t)


def squares(g):
    """
    Count the number of squares for each node
    Parameters
    ----------
    g: igraph Graph object
       The input graph.

    Returns
    -------
    List with N entries (N is number of nodes) that give the number of squares a node is part of.
    """

    cliques = g.cliques(min=4, max=4)
    result = [0] * g.vcount()
    for i, j, k, l in cliques:
        result[i] += 1
        result[j] += 1
        result[k] += 1
        result[l] += 1
    return result


def statistics_square_count(A_in):
    """
    Compute the square count of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Square count
    """

    A_igraph = igraph.Graph.Adjacency((A_in > 0).tolist()).as_undirected()
    return int(np.sum(squares(A_igraph)) / 4)


def statistics_power_law_alpha(A_in):
    """
    Compute the power law coefficient of the degree distribution of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Power law coefficient
    """


    if sp.issparse(A_in):
        degrees = np.array(A_in.sum(axis=1)).flatten()
    else:
        degrees = A_in.sum(axis=0).flatten()
    return powerlaw.Fit(degrees, xmin=max(np.min(degrees),1)).power_law.alpha


def statistics_gini(A_in):
    """
    Compute the Gini coefficient of the degree distribution of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Gini coefficient
    """

    n = A_in.shape[0]
    if sp.issparse(A_in):
        degrees = np.array(A_in.sum(axis=1)).flatten()
    else:
        degrees = A_in.sum(axis=0).flatten()
    degrees_sorted = np.sort(degrees)
    G = (2 * np.sum(np.array([i * degrees_sorted[i] for i in range(len(degrees))]))) / (n * np.sum(degrees)) - (
                                                                                                               n + 1) / n
    return float(G)


def statistics_edge_distribution_entropy(A_in):
    """
    Compute the relative edge distribution entropy of the input graph.

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Rel. edge distribution entropy
    """
    if sp.issparse(A_in):
        degrees = np.array(A_in.sum(axis=1)).flatten()
        m = 0.5 * np.sum(A_in.power(2))
    else:
        degrees = A_in.sum(axis=0).flatten()
        m = 0.5 * np.sum(np.square(A_in))
    n = A_in.shape[0]

    H_er = 1 / np.log(n) * np.sum(-degrees / (2 * float(m)) * np.log((degrees+.0001) / (2 * float(m))))
    return H_er


def statistics_compute_cpl(A):
    """Compute characteristic path length."""
    P = sp.csgraph.shortest_path(sp.csr_matrix(A))
    return P[((1 - np.isinf(P)) * (1 - np.eye(P.shape[0]))).astype(np.bool)].mean()


def compute_graph_statistics(A_in, Z_obs=None):
    """

    Parameters
    ----------
    A_in: sparse matrix
          The input adjacency matrix.
    Z_obs: np.matrix [N, K], where K is the number of classes.
          Matrix whose rows are one-hot vectors indicating the class membership of the respective node.

    Returns
    -------
    Dictionary containing the following statistics:
             * Maximum, minimum, mean degree of nodes
             * Size of the largest connected component (LCC)
             * Wedge count
             * Claw count
             * Triangle count
             * Square count
             * Power law exponent
             * Gini coefficient
             * Relative edge distribution entropy
             * Assortativity
             * Clustering coefficient
             * Number of connected components
             * Intra- and inter-community density (if Z_obs is passed)
             * Characteristic path length
    """
    A = A_in.copy()

    # assert((A == A.T).all())
    # A_graph = nx.from_numpy_matrix(A).to_undirected()

    statistics = {}

    d_max, d_min, d_mean = statistics_degrees(A)

    # Degree statistics
    statistics['d_max'] = d_max
    statistics['d_min'] = d_min
    statistics['d'] = d_mean

    # node number & edger number
    # statistics['node_num'] = A_graph.number_of_nodes()
    # statistics['edge_num'] = A_graph.number_of_edges()

    # largest connected component
    LCC = statistics_LCC(A)

    statistics['LCC'] = LCC.shape[0]
    # # wedge count
    # statistics['wedge_count'] = statistics_wedge_count(A)
    #
    # # claw count
    statistics['claw_count'] = statistics_claw_count(A)

    # triangle count
    statistics['triangle_count'] = statistics_triangle_count(A)

    # Square count
    # statistics['square_count'] = statistics_square_count(A)

    # power law exponent
    statistics['power_law_exp'] = statistics_power_law_alpha(A)

    # gini coefficient
    statistics['gini'] = statistics_gini(A)

    # Relative edge distribution entropy
    statistics['rel_edge_distr_entropy'] = statistics_edge_distribution_entropy(A)

    # Assortativity
    # statistics['assortativity'] = nx.degree_assortativity_coefficient(A_graph)

    # Clustering coefficient
    statistics['clustering_coefficient'] = 3 * statistics['triangle_count'] / (statistics['claw_count']+1)

    # Number of connected components
    # statistics['n_components'] = connected_components(A)[0]

    # if Z_obs is not None:
    #     # inter- and intra-community density
    #     intra, inter = statistics_cluster_props(A, Z_obs)
    #     statistics['intra_community_density'] = intra
    #     statistics['inter_community_density'] = inter

    statistics['cpl'] = statistics_compute_cpl(A)

    return statistics