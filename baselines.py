from deeprobust.graph.global_attack import Random,Metattack,DICE, MinMax,NIPA
from deeprobust.graph.targeted_attack import FGA, Nettack, IGAttack, RLS2V
from deeprobust.graph.utils import *
from utils import change_dataset_format,add_nodes,generate_injected_features,injecting_nodes,select_nodes,single_test
from deeprobust.graph.rl.nipa_env import NodeInjectionEnv, GraphNormTool, StaticGraph, NodeAttackEnv
from tqdm import tqdm
import networkx as nx
# def assign_data

def random_attack(dataset,args):
    features, adj, labels, idx_train,idx_test,_ = change_dataset_format(dataset)
    model = Random()
    n_perturbations = int(args.ptb_rate * (adj.sum() // 2))
    model.attack(adj,n_perturbations)
    modified_adj = model.modified_adj
    modified_adj = normalize_adj(modified_adj)
    modified_adj = sparse_mx_to_torch_sparse_tensor(modified_adj)
    modified_adj = modified_adj.to(args.device)
    return modified_adj

def meta_attack(dataset, defense_model,args):
    features, adj, labels, idx_train, idx_test,_ = change_dataset_format(dataset)
    model = Metattack(model=defense_model,nnodes=adj.shape[0],feature_shape=features.shape,attack_structure=True,attack_features=True,device=args.device)
    model.to(args.device)
    n_perturbations = int(args.ptb_rate * (adj.sum() // 2))
    model.attack(features,adj,labels,idx_train,idx_test,n_perturbations=n_perturbations,ll_constraint=False)
    modified_adj = model.modified_adj.to_sparse()
    modified_features = sparse_mx_to_torch_sparse_tensor(model.modified_features)
    return modified_adj,modified_features

def dice_attack(dataset,args):
    features, adj, labels, idx_train, idx_test,_ = change_dataset_format(dataset)
    model = DICE()
    n_perturbations = int(args.ptb_rate * (adj.sum() // 2))
    model.to(args.device)
    model.attack(adj, labels, n_perturbations)
    modified_adj = model.modified_adj
    modified_adj = normalize_adj(modified_adj)
    modified_adj = sparse_mx_to_torch_sparse_tensor(modified_adj)
    modified_adj = modified_adj.to(args.device)
    return modified_adj

def topology_attack(dataset,defense_model,args):
    features, adj, labels, idx_train, idx_test,_ = change_dataset_format(dataset,process=True,sparse=False)
    model = MinMax(model=defense_model, nnodes=adj.shape[0], loss_type='CE', device=args.device)
    model = model.to(args.device)
    n_perturbations = int(args.ptb_rate * (adj.sum() // 2))
    model.attack(features, adj, labels, idx_train, n_perturbations)
    modified_adj = model.modified_adj
    return modified_adj

def nipa_attack(dataset,defense_model,args_org):
    from deeprobust.graph.rl.nipa_config import args
    injecting_nodes(dataset,args)
    features, adj, labels, idx_train, idx_test, idx_val = change_dataset_format(dataset)
    StaticGraph.graph = nx.from_scipy_sparse_array(adj)
    dict_of_lists = nx.to_dict_of_lists(StaticGraph.graph)
    setattr(defense_model, 'norm_tool', GraphNormTool(normalize=True, gm='gcn', device=args_org.device))
    features, adj, labels, idx_train, idx_test, idx_val = change_dataset_format(dataset,process=True)
    features=features.to(args_org.device)
    adj = adj.to(args_org.device)
    output = defense_model.predict(features, adj)
    labels = torch.LongTensor(labels).to(args_org.device)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    defense_model.eval()
    output = defense_model(defense_model.features, defense_model.adj_norm)
    preds = output.max(1)[1].type_as(labels)
    acc = preds[:len(labels)].eq(labels).double()
    acc_test = acc[idx_test]
    N = dataset.adj.shape[0]
    env = NodeInjectionEnv(features, labels, idx_train, idx_val, dict_of_lists, defense_model, ratio=args.ratio,
                           reward_type=args.reward_type,args=args_org,N=N)
    agent = NIPA(env, features, labels, env.idx_train, idx_val, idx_test, dict_of_lists, num_wrong=0,
                 ratio=args.ratio, reward_type=args.reward_type,
                 batch_size=args.batch_size, save_dir=args.save_dir,
                 bilin_q=args.bilin_q, embed_dim=args.latent_dim,
                 mlp_hidden=args.mlp_hidden, max_lv=args.max_lv,
                 gm=args.gm, device=args_org.device,N=N)
    agent.train(num_episodes=1500, lr=args.learning_rate)
    agent.eval(training=args.phase)



####### The following is a target attack method.

def FGA_attack(dataset,defense_model,args):
    features, adj, labels, idx_train, idx_test, idx_val = change_dataset_format(dataset)
    cnt = 0
    degrees = adj.sum(0).A1
    node_list = select_nodes(defense_model,idx_test,labels)
    num = len(node_list)
    sum_n_perturbations = int(args.ptb_rate * (adj.sum() // 2))
    avg_perturbations = sum_n_perturbations//num
    remaining_purbations = sum_n_perturbations - avg_perturbations*num
    print('=== [Evasion] Attacking %s nodes respectively by %s ===' % (num,args.attack_algorithm))
    modified_adj = None
    model = FGA(defense_model, nnodes=adj.shape[0], device=args.device)
    model.to(args.device)
    for target_node in tqdm(node_list):
        if not args.budget_con:
            n_perturbations = int(degrees[target_node])
        else:
            n_perturbations = avg_perturbations + 1 if remaining_purbations > 0 else avg_perturbations
        remaining_purbations -= 1
        model.attack(features, adj, labels, idx_train, target_node, n_perturbations)
        if modified_adj is None:
            modified_adj = model.modified_adj
        else:
            modified_adj[target_node] = model.modified_adj[target_node]
        acc = single_test(modified_adj, features, target_node,labels,gcn=defense_model)
        if acc == 0:
            cnt += 1
    print('misclassification rate : %s' % (cnt / num))
    modified_adj= sparse_mx_to_torch_sparse_tensor(modified_adj)
    return modified_adj

def nettack_attack(dataset,defense_model,args):
    features, adj, labels, idx_train, idx_test, idx_val = change_dataset_format(dataset)
    cnt = 0
    degrees = adj.sum(0).A1
    node_list = select_nodes(defense_model, idx_test, labels)
    num = len(node_list)
    sum_n_perturbations = int(args.ptb_rate * (adj.sum() // 2))
    avg_perturbations = sum_n_perturbations // num
    remaining_purbations = sum_n_perturbations - avg_perturbations * num
    print('=== [Evasion] Attacking %s nodes respectively by %s ===' % (num,args.attack_algorithm))
    modified_adj = None
    model = Nettack(defense_model, nnodes=adj.shape[0], attack_structure=True, attack_features=False, device=args.device)
    model.to(args.device)
    for target_node in tqdm(node_list):
        if not args.budget_con:
            n_perturbations = int(degrees[target_node])
        else:
            n_perturbations = avg_perturbations + 1 if remaining_purbations > 0 else avg_perturbations
        remaining_purbations -= 1

        model.attack(features, adj, labels, target_node, n_perturbations,verbose=False)
        if modified_adj is None:
            modified_adj = model.modified_adj
            modified_features = model.modified_features
        else:
            modified_adj[target_node] = model.modified_adj[target_node]
            modified_features[target_node] = model.modified_features[target_node]
        acc = single_test(modified_adj, features, target_node, labels, gcn=defense_model)
        if acc == 0:
            cnt += 1
    print('misclassification rate : %s' % (cnt / num))
    modified_adj = sparse_mx_to_torch_sparse_tensor(modified_adj)
    modified_features = sparse_mx_to_torch_sparse_tensor(modified_features)
    return modified_adj,modified_features

def ig_attack(dataset,defense_model,args):
    features, adj, labels, idx_train, idx_test, idx_val = change_dataset_format(dataset)
    cnt = 0
    degrees = adj.sum(0).A1
    node_list = select_nodes(defense_model, idx_test, labels)
    num = len(node_list)
    sum_n_perturbations = int(args.ptb_rate * (adj.sum() // 2))
    avg_perturbations = sum_n_perturbations // num
    remaining_purbations = sum_n_perturbations - avg_perturbations * num
    print('=== [Evasion] Attacking %s nodes respectively by %s ===' % (num, args.attack_algorithm))
    modified_adj = None
    modified_features = None
    model = IGAttack(defense_model, nnodes=adj.shape[0], attack_structure=True, attack_features=True,
                     device=args.device)
    model.to(args.device)
    for target_node in tqdm(node_list):
        if not args.budget_con:
            n_perturbations = int(degrees[target_node])
        else:
            n_perturbations = avg_perturbations + 1 if remaining_purbations > 0 else avg_perturbations
        remaining_purbations -= 1

        model.attack(features, adj, labels, idx_train ,target_node, n_perturbations, steps=20)
        if modified_adj is None:
            modified_adj = model.modified_adj
            modified_features = model.modified_features
        else:
            modified_adj[target_node] = model.modified_adj[target_node]
            modified_features[target_node] = model.modified_features[target_node]

        acc = single_test(modified_adj, features, target_node, labels, gcn=defense_model)
        if acc == 0:
            cnt += 1
    print('misclassification rate : %s' % (cnt / num))
    modified_adj = sparse_mx_to_torch_sparse_tensor(modified_adj)
    modified_features = sparse_mx_to_torch_sparse_tensor(modified_features)
    return modified_adj, modified_features

def s2v_attack(dataset,defense_model,args_org):
    from deeprobust.graph.rl.rl_s2v_config import args
    features, adj, labels, idx_train, idx_test, idx_val = change_dataset_format(dataset)
    dataset.features = normalize_feature(dataset.features)
    StaticGraph.graph = nx.from_scipy_sparse_array(adj)
    dict_of_lists = nx.to_dict_of_lists(StaticGraph.graph)
    setattr(defense_model, 'norm_tool', GraphNormTool(normalize=True, gm='gcn', device=args_org.device))
    features, adj, labels, idx_train, idx_test, idx_val = change_dataset_format(dataset, process=True)
    features = features.to(args_org.device)
    adj = adj.to(args_org.device)
    output = defense_model.predict(features, adj)
    labels = torch.LongTensor(labels).to(args_org.device)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    preds = output.max(1)[1].type_as(labels)[:len(labels)]
    acc = preds.eq(labels).double()
    acc_test = acc[idx_test]
    attack_list = []
    for i in range(len(idx_test)):
        # only attack those misclassifed and degree>0 nodes
        if acc_test[i] > 0 and len(dict_of_lists[idx_test[i]]):
            attack_list.append(idx_test[i])
    # attack_list = select_nodes(defense_model, idx_test, labels)
    if not args.meta_test:
        total = attack_list
        idx_valid = idx_test
    else:
        total = attack_list + idx_val

    acc_test = acc[idx_valid]
    meta_list = []
    num_wrong = 0
    for i in range(len(idx_valid)):
        if acc_test[i] > 0:
            if len(dict_of_lists[idx_valid[i]]):
                meta_list.append(idx_valid[i])
        else:
            num_wrong += 1

    print('meta list ratio:', len(meta_list) / float(len(idx_valid)))
    env = NodeAttackEnv(features, labels, total, dict_of_lists, defense_model, num_mod=args.num_mod,
                        reward_type=args.reward_type)
    agent = RLS2V(env, features, labels, meta_list, attack_list, dict_of_lists, num_wrong=num_wrong,
                  num_mod=args.num_mod, reward_type=args.reward_type,
                  batch_size=args.batch_size, save_dir=args.save_dir,
                  bilin_q=args.bilin_q, embed_dim=args.latent_dim,
                  mlp_hidden=args.mlp_hidden, max_lv=args.max_lv,
                  gm=args.gm, device=args_org.device)
    agent.train(num_steps=args.num_steps, lr=args.learning_rate)
    agent.eval(training=args.phase)

