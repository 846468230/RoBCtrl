import os.path

import random

from center_agent import CenterAgent
from single_agent import Agent
from center_env import ModifiedGraph,NodeAttakEnv
from center_env import eval as evaluate_it
from utils import *
import torch.optim as optim
from torch_geometric.utils import add_self_loops, to_networkx,degree
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import DataParallel
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
# Dpr_data = (changed_data)
from dataset import DataDiffusion
import time
import warnings
from copy import deepcopy
import numpy as np
import random as rd
from utils import draw_heat_map,draw_edge_distribution
from collections import defaultdict
from deeprobust.graph.data import Pyg2Dpr
import pickle as pkl
from collections import Counter
import copy as cp
from tqdm import tqdm
# import sys
# sys.path.append("../")
import gaussian_diffusion as gd
from DNN import DNN
warnings.filterwarnings("ignore")

def RL_attack(dataset, defense_model,args,extra_edges=None):
    def worker_init_fn(worker_id):
        np.random.seed(args.seed + worker_id)
    data = dataset[0]

    if extra_edges is not None:
        data.edge_index = extra_edges

    if args.dataset=="twibot-20":
        split_nums = [42,43]
        # split_nums = [15, 25]
    elif args.dataset == "mgtab-large":
        split_nums = [10,11]
    elif args.dataset == "mgtab":
        split_nums = [530, 531]
    elif args.dataset == "cresci-15":
        split_nums = [3,4]

    in_degree = degree(data.edge_index[0],data.x.size()[0])
    out_degree = degree(data.edge_index[1],data.x.size()[0])
    all_degree = in_degree+out_degree
    train_indices = (dataset.train_index+dataset.val_index).nonzero().squeeze()
    all_indexes = (dataset.train_index + dataset.val_index + dataset.test_index).nonzero().squeeze()
    if args.dataset in ["twibot-20","mgtab-large"]:
        extra_indexes = np.random.choice(range(len(all_indexes),data.x.size()[0]),len(all_indexes),replace=True)
        extra_indexes.sort()
    elif args.dataset in ["mgtab","cresci-15"]:
        extra_indexes = np.random.choice(train_indices,len(train_indices),replace=True)
        extra_indexes.sort()
    if args.use_df:
        diffusion_data = data.x[torch.cat((all_indexes,torch.from_numpy(extra_indexes)))]
        diffusion_loader = DataLoader(diffusion_data, batch_size=args.batch_size, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)
        diffusion,diffusion_model = train_diffusion_model(args,diffusion_loader,data.x.size()[1])
        generate_indice = rd.sample(list(train_indices),50)
        batch_user = data.x[torch.tensor(generate_indice)]
        denoised_user = diffusion.p_sample( diffusion_model.to(args.device),batch_user.to(args.device), args.sampling_steps, args.sampling_noise)
    controled_degrees = all_degree[train_indices]
    degree_map_to_train = {i:train_indices[i] for i in range(len(controled_degrees))}
    sorted_values, sorted_indices = torch.sort(controled_degrees)
    acc_select = defaultdict(list)
    acc_select[1] =[degree_map_to_train[x.item()] for x in sorted_indices[sorted_values <= split_nums[0]]]
    acc_select[2] = [degree_map_to_train[x.item()] for x in sorted_indices[sum(sorted_values <= split_nums[0]): -sum(sorted_values >=split_nums[1])]]
    acc_select[3] = [degree_map_to_train[x.item()] for x in sorted_indices[sorted_values >= split_nums[1]]]

    assert len(acc_select[1])+len(acc_select[2])+len(acc_select[3]) == len(controled_degrees)
    controlled_user = []
    if args.use_agent_num == 1:
        controlled_user.append(rd.sample(acc_select[1], 100))
    elif args.use_agent_num == 2:
        controlled_user.append(rd.sample(acc_select[1], 100))
        # controlled_user.append(rd.sample(acc_select[2], 50))
        controlled_user.append(rd.sample(acc_select[3], 50))
    elif args.use_agent_num == 3:
        assert args.use_df
        controlled_user.append(rd.sample(acc_select[1], 100))
        # controlled_user.append(rd.sample(acc_select[2], 50))
        controlled_user.append(rd.sample(acc_select[3], 75))
        generate_start = data.x.size()[0]
        data.x = torch.cat((data.x,denoised_user.cpu().detach()),)
        controlled_user.append([torch.tensor(generate_start+i) for i in range(len(generate_indice))])
        data.train_mask = index_to_mask(data.train_mask.nonzero(), size=data.x.size(0))
        data.val_mask = index_to_mask(data.val_mask.nonzero() , size=data.x.size(0))
        data.test_mask = index_to_mask(data.test_mask.nonzero(), size=data.x.size(0))


    target_bots = defaultdict(list)
    target_bot_indices = data.y[train_indices].nonzero().squeeze()
    test_target_bot_indices = [ item.item() for item in dataset.test_index.nonzero().squeeze()]
    # test_target_bot_indices = [ item for item in data.y[all_indexes].nonzero().squeeze()  if item in dataset.test_index.nonzero().squeeze()]
    # all_targets = torch.unique(torch.cat((target_bot_indices,test_target_indices)))
    all_targets = target_bot_indices
    for item in all_targets:
        if all_degree[item] <= split_nums[0]:
            target_bots[1].append(item.item())
        elif split_nums[0] < all_degree[item] < split_nums[1]:
            target_bots[2].append(item.item())
        elif split_nums[1] <= all_degree[item]:
            target_bots[3].append(item.item())
    print(len(target_bots[1]),target_bots[2],len(target_bots[3]))
    target_bots_list = []
    target_bots_list.append(target_bots[1])
    target_bots_list.append(target_bots[2])
    target_bots_list.append(target_bots[3])


    target_bots_list = [i for j in target_bots_list for i in j]

    num_accounts = np.where(1 == data.y.data.numpy())[0][-1] + 1
    num_bots = np.where(1 == data.y.data.numpy())[0].size

    args.num_classes = dataset.num_classes
    args.num_features = dataset.num_features

    print(args)

    surrogate_model = defense_model

    ctrled_user_list = sum([user for user in controlled_user], [])

    pred_log, old_label_log, prob_log, correct, loss_test = evaluate_it(surrogate_model, data,
                                                                     target_bots_list,args) # 只会计算给定的对目标机器人判断结果正确率 而且还是训练集

    selected_bots = []
    for prob, bots in zip(prob_log.tolist(), target_bots_list):
        # if prob >= 0.5 and prob <= 0.7:
        selected_bots.append([bots, prob, all_degree[bots]])
    target_bots_list = [i[0] for i in selected_bots]



    succ_list = []
    news_attack_succ = []

    print(f"Attacking Bots # {target_bots_list}...")


    all_new_accounts = [i for i in range(len(ctrled_user_list))]
    all_new_accounts = [ item.item() for item in ctrled_user_list]
    agent_ctrl_accounts = dict()
    start_idx = 0
    end_idx = 0
    for a_id, user in enumerate(controlled_user):
        end_idx = start_idx + len(user)
        agent_ctrl_accounts[a_id] = all_new_accounts[start_idx:end_idx]
        start_idx += len(user)
    q_net_data = deepcopy(data)#.to("cpu")
    new_data = data
    all_labeled_accounts = [ item.item() for item in train_indices] + [ item.item() for item in dataset.test_index.nonzero().squeeze()]
    meta_list, attack_list = target_bots_list, target_bots_list
    # torch.cuda.empty_cache()
    env = NodeAttakEnv(new_data, agent_ctrl_accounts,
                       target_bots_list,all_labeled_accounts, surrogate_model,args=args)

    agents = dict()
    for a_id, accounts in agent_ctrl_accounts.items():
        agents[a_id] = Agent(a_id, env, q_net_data, accounts, meta_list, attack_list,all_labeled_accounts,
                             num_wrong=len(correct) - sum(correct),args=args)

    center_agent = CenterAgent(env, q_net_data, agent_ctrl_accounts, meta_list, attack_list,all_labeled_accounts, agents,
                               num_wrong=len(correct) - sum(correct),args=args)
    if args.num_steps == -1:
        for a_id, agent in center_agent.agents.items():
            agent.net.load_state_dict(torch.load(args.save_dir +f'/{args.dataset}_{args.use_agent_num}_agent_{a_id}_per_{args.rl_train_ptb_rate}_epoch-best.model',map_location=args.map_location))
    else:
        center_agent.train()
    changed_data,temp_edges = center_agent.eval_testsets(test_target_bot_indices)
    # draw_heat_map(dataset,temp_edges,generate_indice,test_target_bot_indices)
    draw_edge_distribution(dataset,temp_edges,generate_indice,test_target_bot_indices,train_indices)
    return changed_data.edge_index, data.x


def train_diffusion_model(args,train_loader,n_item):
    if args.mean_type == 'x0':
        mean_type = gd.ModelMeanType.START_X
    elif args.mean_type == 'eps':
        mean_type = gd.ModelMeanType.EPSILON
    else:
        raise ValueError("Unimplemented mean type %s" % args.mean_type)

    diffusion = gd.GaussianDiffusion(mean_type, args.noise_schedule, args.noise_scale, args.noise_min, args.noise_max, args.steps, args.device).to(args.device)
    out_dims = eval(args.DNN_dims) + [n_item]
    in_dims = out_dims[::-1]
    model = DNN(in_dims, out_dims, args.DNN_emb_size, time_type="cat", norm=args.DNN_norm).to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.df_lr, weight_decay=args.df_weight_decay)
    print("diffusion models ready.")
    param_num = 0
    mlp_num = sum([param.nelement() for param in model.parameters()])
    diff_num = sum([param.nelement() for param in diffusion.parameters()])  # 0
    param_num = mlp_num + diff_num
    print("Number of all diffusion parameters:", param_num)
    best_loss, best_epoch = 100000, 0
    best_test_result = None
    save_path = '{}{}_lr{}_wd{}_bs{}_dims{}_emb{}_{}_steps{}_scale{}_min{}_max{}_sample{}_reweight{}_{}.pth'.format(args.save_path, args.dataset, args.df_lr, args.df_weight_decay, args.df_batch_size, args.DNN_dims,
                                   args.DNN_emb_size, args.mean_type, \
                                   args.steps, args.noise_scale, args.noise_min, args.noise_max, args.sampling_steps,
                                   args.reweight, args.log_name)
    if os.path.exists(save_path):
        model = torch.load(save_path,map_location=args.map_location)
        return diffusion, model
    print("Start training diffusion model...")
    for epoch in range(1, args.df_epochs + 1):
        if epoch - best_epoch >= 20 and epoch > 200:
            print('-' * 18)
            print('Exiting from training early')
            break

        model.train()
        start_time = time.time()

        batch_count = 0
        total_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(args.device)
            batch_count += 1
            optimizer.zero_grad()
            losses = diffusion.training_losses(model, batch, args.reweight)
            loss = losses["loss"].mean()
            total_loss += loss
            loss.backward()
            optimizer.step()

        if epoch % 5 == 0:
            # valid_results = evaluate(test_loader, valid_y_data, train_data, eval(args.topN))
            # if args.tst_w_val:
            #     test_results = evaluate(test_twv_loader, test_y_data, mask_tv, eval(args.topN))
            # else:
            #     test_results = evaluate(test_loader, test_y_data, mask_tv, eval(args.topN))
            # evaluate_utils.print_results(None, valid_results, test_results)

            if total_loss < best_loss:  # recall@20 as selection
                best_loss, best_epoch = total_loss , epoch
                # best_results = valid_results
                # best_test_results = test_results

                if not os.path.exists(args.save_path):
                    os.makedirs(args.save_path)
                torch.save(model,
                           '{}{}_lr{}_wd{}_bs{}_dims{}_emb{}_{}_steps{}_scale{}_min{}_max{}_sample{}_reweight{}_{}.pth' \
                           .format(args.save_path, args.dataset, args.df_lr, args.df_weight_decay, args.df_batch_size, args.DNN_dims,
                                   args.DNN_emb_size, args.mean_type, \
                                   args.steps, args.noise_scale, args.noise_min, args.noise_max, args.sampling_steps,
                                   args.reweight, args.log_name))

        print("Diffusion Runing Epoch {:03d} ".format(epoch) + 'train loss {:.4f}'.format(total_loss) + " costs " + time.strftime(
            "%H: %M: %S", time.gmtime(time.time() - start_time)))
        print('---' * 21)
    print('===' * 21)
    print("Diffusion End. Best Epoch {:03d} ".format(best_epoch))
    # evaluate_utils.print_results(None, best_results, best_test_results)
    print("Diffusion End time: ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    return diffusion, model