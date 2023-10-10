from __future__ import print_function

import numpy as np
import networkx as nx
from tqdm import tqdm
from copy import deepcopy
import random
import pickle as pkl
from collections import defaultdict

import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from q_net_code import QNetNode, NStepQNetNode, node_greedy_actions
from nstep_replay_mem import NstepReplayMem
# from args import args
from deeprobust.graph.defense_pyg import GCN

class CenterAgent(object):
    def __init__(self, env, data, ctrled_accounts, idx_test, idx_train, all_labeled_accounts, agents, num_wrong=0,args=None):
        self.data = data
        self.idx_train = idx_train
        self.idx_test = idx_test
        self.agents = agents
        self.args = args
        self.num_wrong = num_wrong
        self.ctrled_accounts = ctrled_accounts
        self.agent_ctrled_accounts = defaultdict(dict)
        self.all_ctrled_accounts = defaultdict(list)
        self.edge_nums = self.data.edge_index.shape[1]
        for t in all_labeled_accounts:
            for accounts in ctrled_accounts.values():
                self.all_ctrled_accounts[t] += deepcopy(accounts)
        for a_id in self.agents:
            agent_ctrled_accounts = defaultdict(list)
            for t in all_labeled_accounts:
                agent_ctrled_accounts[t] = deepcopy(ctrled_accounts[a_id])
            self.agent_ctrled_accounts[a_id] = agent_ctrled_accounts

        self.mem_pool = NstepReplayMem(memory_size=self.args.mem_size, n_steps=int(self.args.rl_train_ptb_rate * (self.edge_nums//2)),
                                       balance_sample=self.args.reward_type == 'binary')
        self.env = env

        self.eps_start = 1.0
        self.eps_end = 0.05
        self.eps_step = 2000 #100000
        self.burn_in = self.args.burn_in
        self.step = 0
        self.pos = 0
        self.best_eval = None

    def action_agg(self, agent_at, agent_mapping):

        list_at = []
        list_ag = []
        num_agent = range(len(agent_at))
        if len(num_agent) == 3:
            # p = [0.6, 0.3, 0.1]
            # p = [0.34,0.33,0.33]
            p = [0.6,0.3,0.1]
            # p = [0.9,0.05,0.05] 上次是这个
        elif len(num_agent) == 4:
            p = [0.4, 0.3,0.2, 0.1]
        elif len(num_agent) == 1:
            p = [1,]
        elif len(num_agent) == 2:
            p = [0.7,0.3]
        for act in range(len(agent_at[0])):
            agent = np.random.choice(num_agent, p=p)
            
            list_at.append(agent_at[agent][act])
            list_ag.append(agent_mapping[agent])

        return list_at, list_ag

    def run_simulation(self):
        if (self.pos + 1) * self.args.batch_size > len(self.idx_train):
            self.pos = 0
            random.shuffle(self.idx_train)

        selected_idx = self.idx_train[self.pos * self.args.batch_size: (self.pos + 1) * self.args.batch_size]
        self.pos += 1
        self.env.setup(self.idx_train)
        # self.setup(self.idx_train)
        for a_id, accounts in self.agent_ctrled_accounts.items():
            self.agents[a_id].net.list_action_space = deepcopy(accounts)

        t = 0
        list_of_list_st = []
        list_of_list_at = []
        pos = 0
        per_edge_num = int(self.args.rl_train_ptb_rate * (self.edge_nums//2))
        while not self.env.isTerminal():
            # target_node = self.idx_train[t % len(self.idx_train)]
            # print(f"step: {self.step}  timestamp: {t}",end='\r')
            if per_edge_num - t < len(self.idx_train) :
                end = pos + per_edge_num-t
            else:
                end = pos + len(self.idx_train)
            target_nodes = self.idx_train[pos: end]
            agent_at = []
            timestamps = range(t,t + len(target_nodes))
            for a_id, agent in self.agents.items():
                action = agent.make_actions(timestamps, target_nodes)
                agent_at.append(action)

            # aggregate action via sampling
            list_at, list_ag = self.action_agg(
                agent_at, list(self.agents.keys()))

            # # update the action space
            # for agent, target, act in zip(list_ag, self.idx_train, list_at):
            #     self.agents[agent].env.list_action_space[agent][target].remove(
            #         act)
            #     # self.agents[agent].net.list_action_space[target].remove(act) # 这里取消注释后 q_net 那里对于重复动作的判定就可以取消了
            for agent, target, act in zip(list_ag, target_nodes, list_at):
                self.agents[agent].env.list_action_space[agent][target].remove(act)
                self.agents[agent].net.list_action_space[target].remove(act)
            list_st = self.env.cloneState(target_nodes,timestamps)

            # print(f'simu_states: {list_st}')
            # print(f'simu_action: {list_at}')
            # exit()
            first_nodes = target_nodes
            self.env.step(list_at,first_nodes=first_nodes,start=pos)
            # assert (self.env.rewards is not None) == self.env.isTerminal()
            if self.env.isTerminal():
                # rewards = self.env.rewards
                # s_prime = None
                s_prime = self.env.cloneState(target_nodes, timestamps)
                s_prime[-1] = None
                terminal = [False] * (len(list_at)-1) + [True,]
                # rewards = np.zeros(len(list_at), dtype=np.float32)
                # rewards[-1] = self.env.rewards
                rewards = self.env.rewards[:len(list_at)]
            else:
                # rewards = np.zeros(len(list_at), dtype=np.float32)
                rewards = self.env.rewards[:len(list_at)]
                s_prime = self.env.cloneState(target_nodes,timestamps)
                terminal = [False] * len(list_at)

            self.mem_pool.add_list(list_st, list_at, rewards, s_prime,terminal, timestamps)
            list_of_list_st.append(deepcopy(list_st))
            list_of_list_at.append(deepcopy(list_at))

            pos = 0
            t += len(target_nodes)

        if self.args.reward_type == 'nll':
            return
        T = t
        cands = self.env.sample_pos_rewards(len(selected_idx))
        if len(cands):
            for c in cands:
                sample_idx, target = c
                doable = True
                for t in range(T):
                    if self.all_ctrled_accounts is not None and (
                            not list_of_list_at[t][sample_idx] in self.all_ctrled_accounts[target]):
                        doable = False
                        break
                if not doable:
                    continue
                for t in range(T):
                    s_t = list_of_list_st[t][sample_idx]
                    a_t = list_of_list_at[t][sample_idx]
                    s_t = [target, deepcopy(s_t[1])]
                    if t + 1 == T:
                        s_prime = (None, None)
                        r = 1.0
                        term = True
                    else:
                        s_prime = list_of_list_st[t + 1][sample_idx]
                        s_prime = [target, deepcopy(s_prime[1])]
                        r = 0.0
                        term = False
                    self.mem_pool.mem_cells[t].add(s_t, a_t, r, s_prime, term)

    def eval(self):
        self.env.setup(self.idx_train)
        for a_id, agent in self.agents.items():
            agent.net.list_action_space = deepcopy(
                self.agent_ctrled_accounts[a_id])
        t = 0

        cost = defaultdict(list)
        pos = 0
        per_edge_num =int(self.args.rl_train_ptb_rate * (self.edge_nums//2))
        while not self.env.isTerminal():
            agent_at = []
            if per_edge_num - t < len(self.idx_train):
                end = pos + per_edge_num - t
            else:
                end = pos + len(self.idx_train)
            target_nodes = self.idx_train[pos: end]
            timestamps = range(t, t + len(target_nodes))
            for idx, agent in self.agents.items():
                agent_at.append(agent.make_actions(
                    timestamps, target_nodes, greedy=True))

            # aggregate action via sampling
            list_at, list_ag = self.action_agg(
                agent_at, list(self.agents.keys()))

            # print(list_at)
            for action in list_at:
                if action < 100 and action not in cost[0]:
                    cost[0].append(action)
                elif action >= 100 and action < 150 and action not in cost[1]:
                    cost[1].append(action)
                elif action >= 150 and action not in cost[2]:
                    cost[2].append(action)
            # update the action space
            # for agent, target, act in zip(list_ag, self.idx_test, list_at):
                # print(agent, target, act)
                # self.agents[agent].env.list_action_space[agent][target].remove(act)
                # self.agents[agent].net.list_action_space[target].remove(act)
            for agent, target, act in zip(list_ag, target_nodes, list_at):
                self.agents[agent].env.list_action_space[agent][target].remove(act)
                self.agents[agent].net.list_action_space[target].remove(act)
            # print(list_at)
            first_nodes = target_nodes
            self.env.step(list_at,first_nodes=first_nodes ,start=pos,eval_flag=True)
            t += len(target_nodes)

        # print(f"cost: {[len(a) for a in cost.values()]}")
        acc = 1 - (self.env.binary_rewards + 1.0) / 2.0
        acc = np.sum(acc) / len(self.idx_train)
        # for i in range(len(self.idx_train)):
        #     for e in range(self.env.modified_list[i].added_edges.shape[1]):
        #         print(
        #             f'({self.env.modified_list[i].added_edges[0, e]} {self.env.modified_list[i].added_edges[1, e]})')
        print('\033[93m trained target bot average test: acc %.5f\033[0m' % (acc))
        torch.cuda.empty_cache()
        if self.args.phase == 'train':
            if self.best_eval is None or acc < self.best_eval:
                print(
                    '----saving to best attacker since this is the best attack rate so far.----')
                for a_id, agent in self.agents.items():
                    torch.save(agent.net.state_dict(), self.args.save_dir +
                               f'/{self.args.dataset}_{self.args.use_agent_num}_agent_{a_id}_per_{self.args.rl_train_ptb_rate}_epoch-best.model')
                with open(self.args.save_dir + f'/{self.args.dataset}per_{self.args.rl_train_ptb_rate}_epoch-best.txt', 'w') as f:
                    f.write('%.4f\n' % acc)
                with open(self.args.save_dir + f'/{self.args.dataset}per_{self.args.rl_train_ptb_rate}_attack_solution.txt', 'w') as f:
                    for i in range(len(self.idx_train)):
                        f.write('%d: [' % self.idx_train[i])
                        for e in range(self.env.modified_list[i].added_edges.shape[1]):
                            f.write(
                                f'({self.env.modified_list[i].added_edges[0, e]} {self.env.modified_list[i].added_edges[1, e]})')
                        f.write('] succ: %d\n' % (self.env.binary_rewards[i]))
                self.best_eval = acc
        return acc

    def eval_testsets(self,test_bots):
        self.env.setup(test_bots)
        self.env.test_phase = True
        for a_id, agent in self.agents.items():
            agent.net.list_action_space = deepcopy(
                self.agent_ctrled_accounts[a_id])
        t = 0
        pos = 0
        cost = defaultdict(list)
        per_edge_num = int(self.args.ptb_rate * (self.edge_nums//2))
        while not self.env.isTerminal():
            agent_at = []
            if per_edge_num - t < len(test_bots):
                end = pos + per_edge_num - t
            else:
                end = pos + len(test_bots)
            target_nodes = test_bots[pos: end]
            timestamps = range(t, t + len(target_nodes))
            for idx, agent in self.agents.items():
                agent_at.append(agent.make_actions(
                    timestamps, target_nodes, greedy=True))

            # aggregate action via sampling
            list_at, list_ag = self.action_agg(
                agent_at, list(self.agents.keys()))

            # print(list_at)
            for action in list_at:
                if action < 100 and action not in cost[0]:
                    cost[0].append(action)
                elif action >= 100 and action < 150 and action not in cost[1]:
                    cost[1].append(action)
                elif action >= 150 and action not in cost[2]:
                    cost[2].append(action)
            # update the action space
            # for agent, target, act in zip(list_ag, self.idx_test, list_at):
                # print(agent, target, act)
                # self.agents[agent].env.list_action_space[agent][target].remove(act)
                # self.agents[agent].net.list_action_space[target].remove(act)
            for agent, target, act in zip(list_ag, target_nodes, list_at):
                self.agents[agent].env.list_action_space[agent][target].remove(act)
                self.agents[agent].net.list_action_space[target].remove(act)

            # print(list_at)
            first_nodes = target_nodes
            self.env.step(list_at, first_nodes = first_nodes,start=pos ,eval_flag=True,reverse=True)
            t += len(target_nodes)

        # print(f"cost: {[len(a) for a in cost.values()]}")
        acc = 1 - (self.env.binary_rewards + 1.0) / 2.0
        acc = np.sum(acc) / len(test_bots)
        # for i in range(len(self.idx_train)):
        #     for e in range(self.env.modified_list[i].added_edges.shape[1]):
        #         print(
        #             f'({self.env.modified_list[i].added_edges[0, e]} {self.env.modified_list[i].added_edges[1, e]})')
        print('\033[93m test target bot average test: acc %.5f\033[0m' % (acc))
        temp_edges = []
        for i in range(len(test_bots)):
            new_edges = self.env.modified_list[i].get_new_edges()
            # print(temp_new_data.edge_index)
            temp_edges.append(new_edges)
            self.env.static_data.edge_index = torch.cat([self.env.static_data.edge_index, new_edges.to(self.args.device)], dim=1)
        return self.env.static_data, temp_edges

    def setup(self,target_nodes):
        self.mem_pool.n_steps = int(self.args.ptb_rate * self.data.edge_index.shape[1] / len(target_nodes))
        for agent in self.agents.values():
            agent.mem_pool.n_steps = int(self.args.ptb_rate * self.data.edge_index.shape[1] / len(target_nodes))

    def train(self):
        pbar = range(self.burn_in)
        for p in pbar:
            self.run_simulation()

        # print("-"*10 + f"\ncheck one: {self.env.modified_list[0].modified}\n" + "-"*10 + "\n")

        pbar = tqdm(range(self.args.num_steps), unit='step')
        self.optimizer = dict()
        for a_id, agent in self.agents.items():
            self.optimizer[a_id] = optim.Adam(
                agent.net.parameters(), lr=self.args.rl_lr)

        self.loss_log = defaultdict(list)
        self.qval_log = defaultdict(list)

        for self.step in pbar:

            self.run_simulation()

            # print("-"*10 + f"\ncheck two: {self.env.modified_list[0].modified}\n" + "-"*10 + "\n")

            if self.step % 50 == 0:
                for agent in self.agents.values():
                    agent.take_snapshot()
                # print(f'loss: {self.loss_log}')
                # print(f'qval: {self.qval_log}')
            if self.step % 100 == 0 and self.step > 0:
                acc = self.eval()
                # if acc != 1:
                #     break
            # if self.step < 10:
            #     continue
            for train_time in range(self.args.max_timesteps):
                cur_time, list_st, list_at, list_rt, list_s_primes, list_term = self.mem_pool.sample(
                batch_size=self.args.rl_batch_size)

                # print(list_at)
             # print(list_rt)
                mem_agent_mapping = defaultdict(list)
                for i, a in enumerate(list_at):
                    for a_id, accounts in self.ctrled_accounts.items():
                        if a in accounts:
                         mem_agent_mapping[a_id].append(i)

                agent_list_target = dict()
                for a_id, mem_id in mem_agent_mapping.items():
                    list_target = torch.Tensor([list_rt[idx] for idx in mem_id])
                    if self.args.device != 'cpu':
                        list_target = list_target.to(self.args.device)
                    agent_list_target[a_id] = list_target

                if not list_term[0]:
                    for a_id, mem_id in mem_agent_mapping.items():
                        selected_s_primes = [list_s_primes[idx] for idx in mem_id]
                        target_nodes, _ = zip(*selected_s_primes)
                        _, q_t_plus_1 = self.agents[a_id].old_net(cur_time + 1, selected_s_primes, None)
                        _, q_rhs = node_greedy_actions(target_nodes, q_t_plus_1, self.agents[a_id].old_net)
                        agent_list_target[a_id] += self.args.discount_factor * q_rhs

                # print(list_target.shape)
                for a_id, list_target in agent_list_target.items():
                    agent_list_target[a_id] = Variable(list_target.view(-1, 1))

                # print(list_target)

                # exit()
                for a_id, mem_id in mem_agent_mapping.items():
                    agent = self.agents[a_id]
                    agent.net.list_action_space = deepcopy(self.agent_ctrled_accounts[a_id])
                    selected_list_st = [list_st[idx] for idx in mem_id]
                    selected_list_at = [list_at[idx] for idx in mem_id]
                    _, q_sa = agent.net(cur_time, selected_list_st, selected_list_at)
                    q_sa = torch.cat(q_sa, dim=0)
                    # print(q_sa)
                    loss = F.mse_loss(q_sa, agent_list_target[a_id])
                    self.optimizer[a_id].zero_grad()
                    loss.backward()
                    # print(self.net.parameters())
                    for param in agent.net.parameters():
                        # print(param)
                        if param.grad is not None:
                            param.grad.data.clamp_(-1, 1)
                    # exit()
                    self.optimizer[a_id].step()
                    pbar.set_description('loss: %0.5f, q_val: %.5f' % (loss.detach().cpu().numpy(),
                                                                   torch.mean(q_sa).detach().cpu().numpy()))
                    self.loss_log[a_id].append(loss.detach().cpu().numpy().tolist())
                    self.qval_log[a_id].append(torch.mean(q_sa).detach().cpu().numpy().tolist())

        pkl.dump(self.loss_log, open(self.args.save_dir +
                 f'/q_loss_{self.args.dataset}_per_{self.args.rl_train_ptb_rate}_{self.args.defense_algorithm}.pkl', 'wb'))
        pkl.dump(self.qval_log, open(self.args.save_dir +
                 f'/q_val_{self.args.dataset}_per_{self.args.rl_train_ptb_rate}_{self.args.defense_algorithm}.pkl', 'wb'))
