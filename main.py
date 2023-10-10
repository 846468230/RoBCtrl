import os
import sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'
os.environ['NUMEXPR_MAX_THREADS'] = '32'
import torch
import numpy as np
import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    # np.random.seed(seed)
setup_seed(20)
import argparse
from copy import deepcopy
import time
import logging
from deeprobust.graph.data import Dataset,Dpr2Pyg, Pyg2Dpr
from deeprobust.graph.defense import GCN
from defense_methods import GCN_defense, AirGNN_defense, SAGE_defense,SAGE,ProGNN_defense,GAT_defense,APPNP_defense,GPRGNN_defense,SGC_defense,ARMA_defense
from baselines import random_attack, meta_attack, dice_attack, topology_attack, nipa_attack,FGA_attack,nettack_attack,ig_attack,s2v_attack
from utils import get_device,change_dataset_format,normalize_adj,sparse_mx_to_torch_sparse_tensor,torch_sparse_tensor_to_sparse_mx
from dataset import SocialBotDataset,index_to_mask
from attack import RL_attack
from statistics_utils import compute_graph_statistics,cal_attacked
from pprint import pprint
from torch_geometric.utils import to_scipy_sparse_matrix

def printing_opt(opt):
    return "\n".join(["%15s | %s" % (e[0], e[1]) for e in sorted(vars(opt).items(), key=lambda x: x[0])])
parser = argparse.ArgumentParser(description='Attack')
parser.add_argument('--dataset', type=str, default="mgtab", choices=["citeseer","twibot-20","mgtab","mgtab-large","cresci-15"])
parser.add_argument('--redirect', type=bool, default=False)
#{"cresci-2015":0,"botometer-feedback-2019":1,"cresci-rtbust-2019":2,"vendor-purchased-2019":4,"varol-2017":5}
parser.add_argument('--folds', type=int, default=10)

parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument("--rl_lr", type=float, default=0.01, help="rl learning rate")
parser.add_argument("--rl_weight_decay", type=float,default=0.01, help="rl weight decay")
parser.add_argument("--rl_batch_size", type=int, default=32, help="batch size")
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--model', type=str, default='AirGNN')
parser.add_argument('--lambda_amp', type=float, default=0)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--max_timesteps', type=int, default=1)
parser.add_argument('--max_episodes', type=int, default=100)

parser.add_argument("--mem_size", type=int, default=1000,help="replay memory cell size")
parser.add_argument("--nhop", type=int, default=2, help="number of hops")
parser.add_argument("--num_steps", type=int, default=-1,help="agent training step,-1 will load the trained agent")
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--use_agent_num', type=int, default=3, choices=[1,2,3],help = "number agents are used to control attacks ")
parser.add_argument("--reward_type", type=str,default="nll", help="nll or binary")
parser.add_argument("--num_mod", type=int, default=5, help="budget")
parser.add_argument('--replay_memory_init_size', type=int, default=500)
parser.add_argument('--update_target_estimator_every', type=int, default=1)
parser.add_argument("--save_dir", type=str, default="attack_log",help="the attack logging directory")
parser.add_argument("--device", type=int, default=2, choices=[-1,0,1,2,3,4,5,6,7], help="run device (cpu | cuda)")
parser.add_argument('--layer_num', type=int, default=2)
parser.add_argument('--width_num',type=int,default=2)
parser.add_argument('--discount_factor', type=float, default=0.99)
parser.add_argument('--epsilon_start', type=float, default=1.)
parser.add_argument('--epsilon_end', type=float, default=0.1)
parser.add_argument('--epsilon_decay_steps', type=int, default=100)
parser.add_argument('--norm_step', type=int, default=200)
parser.add_argument('--mlp_layers', type=list, default=[64, 128, 256, 128, 64])
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--hid_dim', type=int, default=64)
parser.add_argument("--mlp_hidden", type=int, default=64,help="hidden layer dimension for MLP in Q net")
parser.add_argument("--max_lv", type=int, default=2,help="max pooling layers for Q net")
parser.add_argument("--latent_dim", type=int, default=128,help="hidden layer dimension for Q net")
parser.add_argument("--burn_in", type=int, default=1, help="burn in steps")
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--with_bn', type=int, default=0)
parser.add_argument("--attack_algorithm", type=str, default="rl",choices=["random","mettack","dice","topology","nipa","fga","nettack","ig","s2v","rl"])
parser.add_argument("--defense_algorithm", type=str, default="GCN",choices=["GCN","GAT","SAGE","AirGNN","APPNP","GPRGNN","ProGNN","SGC","ARMA"])
parser.add_argument('--ptb_rate', type=float, default=0.01,  help='pertubation rate') # twi20 0.04 mgtab-large  0.04 0.030 mgtab 0.01  0.0035 target cresci-15 0.15
parser.add_argument('--rl_train_ptb_rate', type=float, default=0.01,  help='pertubation rate') # twi20 0.04 mgtab-large  0.04 0.030 mgtab 0.01 0.007 cresci-15 0.72
parser.add_argument('--budget_con', type=bool, default=True)
parser.add_argument("--phase", type=str, default="train",help="rl model phase, train or test")
parser.add_argument("--bilin_q", type=bool, default=True,help="whether using bilinear Q function")
parser.add_argument('--seed', type=int, default=20)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--out_dim', type=int, default=2)
parser.add_argument('--symmetric', action='store_true', default=False,help='whether use symmetric matrix')
# params for diffusion and DNN
parser.add_argument('--use_df', type=bool, default=1, help='user_diffusion')
parser.add_argument('--df_lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--df_weight_decay', type=float, default=0.0)
parser.add_argument('--df_batch_size', type=int, default=400)
parser.add_argument('--df_epochs', type=int, default=1000, help='upper epoch limit')
parser.add_argument('--save_path', type=str, default='./saved_models/', help='save model path')
parser.add_argument('--log_name', type=str, default='log', help='the log name')
parser.add_argument('--time_type', type=str, default='cat', help='cat or add')
parser.add_argument('--DNN_dims', type=str, default='[1000]', help='the dims for the DNN')
parser.add_argument('--DNN_norm', type=bool, default=False, help='Normalize the input or not')
parser.add_argument('--DNN_emb_size', type=int, default=10, help='timestep embedding size')
# params for diffusion
parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
parser.add_argument('--steps', type=int, default=200, help='diffusion steps')
parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
parser.add_argument('--noise_scale', type=float, default=0.1, help='noise scale for noise generating')
parser.add_argument('--noise_min', type=float, default=0.0001, help='noise lower bound for noise generating')
parser.add_argument('--noise_max', type=float, default=0.02, help='noise upper bound for noise generating')
parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
parser.add_argument('--sampling_steps', type=int, default=0, help='steps of the forward process during inference')
parser.add_argument('--reweight', type=bool, default=True, help='assign different weight to different timestep or not')
parser.add_argument('--cal_graph', type=bool, default=False, help='Attribute data for a statistical graph')

attack_feature_methods = ["nettack","ig","rl"]
args = parser.parse_args()
args.device = get_device(args.device)
args.surrogate_path = os.path.join(os.path.curdir,'data',args.dataset+f"_{args.seed}_"+"surrogate.model")
args.save_dir = os.path.join(os.path.curdir , "attack_log")
args.base_path = os.path.join(os.path.curdir,'data')
args.map_location={'cuda:6':'cuda:0','cuda:3':'cuda:0','cuda:1':'cuda:0','cuda:2':'cuda:0','cuda:4':'cuda:0','cuda:7':'cuda:0','cuda:5':'cuda:0'}
if args.attack_algorithm !='rl':
    args.atacked_adj_path = os.path.join(os.path.curdir,"tensors",f"{args.seed}_{args.attack_algorithm}_{args.dataset}_{args.ptb_rate}_adj.pt")
    args.atacked_features_path = os.path.join(os.path.curdir,"tensors",f"{args.seed}_{args.attack_algorithm}_{args.dataset}_{args.ptb_rate}_feature.pt")
else:
    args.atacked_adj_path = os.path.join(os.path.curdir, "tensors",
                                         f"{args.seed}_{args.attack_algorithm}_{args.dataset}_{args.ptb_rate}_adj_df_{args.use_df}.pt")
    args.atacked_features_path = os.path.join(os.path.curdir, "tensors",
                                              f"{args.seed}_{args.attack_algorithm}_{args.dataset}_{args.ptb_rate}_feature_df_{args.use_df}.pt")
if args.redirect:
    if args.general:
        sys.stdout = open(f'{args.dataset}{args.test_dataset}.txt', "w")
    else:
        sys.stdout = open(f'{args.dataset}{args.width_num}{args.layer_num}.txt', "w")

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logging.info("\n" + printing_opt(args))
logger = logging
print("=" * 80)
print("Summary of training process:")
# print("Algorithm    : {}".format(args.algorithm))
print("Batch size   : {}".format(args.batch_size))
print("Dataset      : {}".format(args.dataset))

print(f"Attacking with {args.layer_num} layers and {args.width_num} width")


def simulation(args,data,ori_pyg):
    if args.cal_graph:
        ori_statistical = compute_graph_statistics(data.adj.tocoo())#(data.adj.toarray())
        pprint(ori_statistical)
    features, adj, labels, idx_train, idx_test, idx_val = change_dataset_format(data)
    pyg_data = Dpr2Pyg(data)
    pyg_data = pyg_data[0]
    if args.defense_algorithm in ["GCN","AirGNN","SAGE","ProGNN","GAT","APPNP","GPRGNN","SGC","ARMA"]:
        if os.path.exists(args.surrogate_path):
            surrogate = torch.load(args.surrogate_path,map_location=torch.device('cpu'))
            surrogate = surrogate.to(args.device)
            surrogate.adj_norm = surrogate.adj_norm.to(args.device)
            surrogate.features = surrogate.features.to(args.device)
            surrogate.labels = surrogate.labels.to(args.device)
            surrogate.output = surrogate.output.to(args.device)
        else:
            surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1, nhid=args.hid_dim, dropout=0.5, with_bias=True, weight_decay=args.weight_decay, device=args.device)
            surrogate = surrogate.to(args.device)
            surrogate.fit(features, adj, labels, idx_train, train_iters=args.epochs)
            torch.save(surrogate,args.surrogate_path)
    surrogate.device = args.device
    if args.cal_graph:
        cal_attacked(RL_attack, ori_pyg, surrogate, args, pyg_data.edge_index,data=data)
    # elif args.defense_algorithm == "SAGE":
    #     surrogate = SAGE(features.shape[1], 32, max(labels).item()+1, num_layers=5,
    #             dropout=0.0, lr=0.01, weight_decay=0, device=args.device).to(args.device)
    #     surrogate.fit(pyg_data, train_iters=1000, patience=1000, verbose=True)
    print(f'=== {args.attack_algorithm} is attacking the original graph ===')
    if not args.train:
        modified_adj = torch.load(args.atacked_adj_path,map_location=args.map_location)
        if args.attack_algorithm in attack_feature_methods:
            modified_features = torch.load(args.atacked_features_path,map_location=args.map_location)
    else:
        start = time.time()
        if args.attack_algorithm == "random":
            modified_adj = random_attack(data, args)
        elif args.attack_algorithm == "mettack":
            modified_adj, modified_features = meta_attack(data, surrogate, args)
            modified_features.to(args.device)
        elif args.attack_algorithm == "dice":
            modified_adj = dice_attack(data, args)
        elif args.attack_algorithm == "topology":
            modified_adj = topology_attack(data, surrogate, args)
        elif args.attack_algorithm == "nipa":
            nipa_attack(data, surrogate, args)
            return
        elif args.attack_algorithm == "fga":
            modified_adj = FGA_attack(data, surrogate, args)
        elif args.attack_algorithm == "nettack":
            modified_adj, modified_features = nettack_attack(data, surrogate, args)
            modified_features.to(args.device)
        elif args.attack_algorithm == "ig":
            modified_adj, modified_features = ig_attack(data, surrogate, args)
            modified_features.to(args.device)
        elif args.attack_algorithm == "s2v":
            s2v_attack(data, surrogate, args)
        elif args.attack_algorithm == "rl":
            modified_adj,modified_features = RL_attack(ori_pyg,surrogate,args,pyg_data.edge_index) # format: edge_index
            modified_features.to(args.device)
        end = time.time()
        print(f"attack total time spent: {end - start}")
        torch.save(modified_adj,args.atacked_adj_path)
        if args.attack_algorithm in attack_feature_methods:
            torch.save(modified_features,args.atacked_features_path)
    features, adj, labels, idx_train, idx_test,_ = change_dataset_format(data, process=True)
    features = features.to(args.device)
    adj = adj.to(args.device)
    labels = labels.to(args.device)
    print(f'=== testing {args.defense_algorithm} on original(clean) graph ===')

    if args.defense_algorithm == "GCN":
        # GCN_defense(features, adj, labels, idx_train, idx_test, args.hid_dim, args.device, epochs=args.epochs, verbose=True)
        GCN_defense(pyg_data, args.hid_dim, args.device, epochs=args.epochs, args=args)
    elif args.defense_algorithm == "AirGNN":
        AirGNN_defense(pyg_data,args.hid_dim, args.device, epochs=args.epochs, args=args)
    elif args.defense_algorithm == "SAGE":
        SAGE_defense(pyg_data, args.hid_dim, args.device, epochs=args.epochs, args=args)
    elif args.defense_algorithm == "ProGNN":
        ProGNN_defense(features, adj, labels, idx_train,idx_val, idx_test, args.hid_dim, args.device, epochs=args.epochs, verbose=True,args=args)
    elif args.defense_algorithm == "GAT":
        GAT_defense(pyg_data, args.hid_dim, args.device, epochs=args.epochs, args=args)
    elif args.defense_algorithm == "APPNP":
        APPNP_defense(pyg_data, args.hid_dim, args.device, epochs=args.epochs, args=args)
    elif args.defense_algorithm == "GPRGNN":
        GPRGNN_defense(pyg_data, args.hid_dim, args.device, epochs=args.epochs, args=args)
    elif args.defense_algorithm == "SGC":
        SGC_defense(pyg_data, args.hid_dim, args.device, epochs=args.epochs, args=args)
    elif args.defense_algorithm == "ARMA":
        ARMA_defense(pyg_data, args.hid_dim, args.device, epochs=args.epochs, args=args)
    if args.attack_algorithm in attack_feature_methods:
        features = modified_features

    print(f'=== testing {args.defense_algorithm} on perturbed graph after attacked by {args.attack_algorithm} ===')
    modified_adj = modified_adj.to(args.device)
    if args.attack_algorithm != "rl":
        data.features = torch_sparse_tensor_to_sparse_mx(features.to(args.device))
        data.adj =  torch_sparse_tensor_to_sparse_mx(modified_adj.to(args.device))
        pyg_data = Dpr2Pyg(data)
        pyg_data = pyg_data[0]

    else:
        pyg_data = Dpr2Pyg(data)
        pyg_data = pyg_data[0]
        pyg_data.edge_index = modified_adj
        pyg_data.x = features.to(args.device)
        pyg_data.train_mask = index_to_mask(data.idx_train, size=features.size(0))
        pyg_data.val_mask = index_to_mask(data.idx_val, size=features.size(0))
        pyg_data.test_mask = index_to_mask(data.idx_test, size=features.size(0))
    if args.defense_algorithm == "GCN":
        # GCN_defense(features, modified_adj, labels, idx_train, idx_test, args.hid_dim, args.device, epochs=args.epochs,verbose=True)
        GCN_defense(pyg_data, args.hid_dim, args.device, epochs=args.epochs, args=args,attacked=True)
    elif args.defense_algorithm == "AirGNN":
        AirGNN_defense(pyg_data,args.hid_dim, args.device, epochs=args.epochs, args=args,attacked=True)
    elif args.defense_algorithm == "SAGE":
        SAGE_defense(pyg_data,args.hid_dim, args.device, epochs=args.epochs, args=args,attacked=True)
    elif args.defense_algorithm == "ProGNN":
        ProGNN_defense(features, modified_adj, labels, idx_train, idx_val,idx_test, args.hid_dim, args.device, epochs=args.epochs,
                       verbose=True, args=args,attacked=True)
    elif args.defense_algorithm == "GAT":
        GAT_defense(pyg_data, args.hid_dim, args.device, epochs=args.epochs, args=args,attacked=True)
    elif args.defense_algorithm == "APPNP":
        APPNP_defense(pyg_data, args.hid_dim, args.device, epochs=args.epochs, args=args,attacked=True)
    elif args.defense_algorithm == "GPRGNN":
        GPRGNN_defense(pyg_data, args.hid_dim, args.device, epochs=args.epochs, args=args,attacked=True)
    elif args.defense_algorithm == "SGC":
        SGC_defense(pyg_data, args.hid_dim, args.device, epochs=args.epochs, args=args,attacked=True)
    elif args.defense_algorithm == "ARMA":
        ARMA_defense(pyg_data, args.hid_dim, args.device, epochs=args.epochs, args=args,attacked=False)


def main(args):
    start = time.time()
    if args.dataset in ["twibot-20","mgtab","mgtab-large","cresci-15"]:
        ori_data = SocialBotDataset(root="./data",dataset=args.dataset)
        data = Pyg2Dpr(ori_data)
        # data.x = data.x.to(args.device)
        # data.y = data.y.to(args.device)
        rows, cols = data.adj.nonzero()
        data.adj[rows, cols] = 1
        data.adj[cols, rows] = data.adj[rows, cols]   # change the input graph into a symmetric
        def check_adj(adj):
            """Check if the modified adjacency is symmetric and unweighted.
            """
            assert np.abs(adj - adj.T).sum() == 0, "Input graph is not symmetric"
            assert adj.tocsr().max() == 1, "Max value should be 1!"
            assert adj.tocsr().min() == 0, "Min value should be 0!"

        check_adj(data.adj.tolil())
    else:
        data = Dataset(root='/tmp/', name=args.dataset)
        ori_data = None
    end = time.time()
    print(f"ini time: {end - start}")
    simulation(args,data,ori_data)

    return


if __name__ == '__main__':
    main(args)
