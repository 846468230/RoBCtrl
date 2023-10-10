import numpy as np
import random as rd
from copy import deepcopy
from tqdm import tqdm
from collections import defaultdict
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from utils import sparse_mx_to_torch_sparse_tensor
from utils import index_to_mask
from deeprobust.graph.defense_pyg import GCN
# from args import args


def eval(model, data, target_news_list,args):
	"""
	Evaluate the attack performance given target news node index
	"""

	target_news_index = torch.tensor(target_news_list, dtype=torch.long)
	data.target_mask = index_to_mask(target_news_index, data.num_nodes)

	out_log = []
	model.eval()
	with torch.no_grad():
		data = data.to(args.device)
		n = data.x.size()[0]
		adj = sp.csr_matrix((np.ones(data.edge_index.shape[1]),
							 (data.edge_index[0].cpu(), data.edge_index[1].cpu())), shape=(n, n))
		# features = sp.csr_matrix(data.x, dtype=np.float32)  # pyg_data.x.numpy()
		adj = sparse_mx_to_torch_sparse_tensor(adj).to(args.device)
		# features = sparse_mx_to_torch_sparse_tensor(features)
		out = model(data.x,adj)[data.target_mask]
		y = data.y[target_news_index]
		out_log = [F.softmax(out, dim=1), y] # [F.softmax(out, dim=1), y] [out, y]
		loss_test = F.nll_loss(out, y, reduction="none").data.cpu().numpy().tolist()

	pred_log, label_log, prob_log = [], [], []

	pred_y, y = out_log[0].data.cpu().numpy().argmax(
		axis=1), out_log[1].data.cpu().numpy().tolist()
	prob_log.extend(out_log[0].data.cpu().numpy()[:, 1].tolist())
	pred_log.extend(pred_y)
	label_log.extend(y)

	pred_log, label_log, prob_log = np.array(
		pred_log), np.array(label_log), np.array(prob_log)
	correct = (label_log == pred_log).astype(int).tolist()

	return pred_log, label_log, prob_log, correct, loss_test


class ModifiedGraph(object):
	def __init__(self):
		self.added_edges = None
		# self.new_data = deepcopy(data.to("cpu"))
		self.modified = False

	def add_edge(self, x: int, y: int):
		"""
		x: news node
		y: account node
		"""
		# new_edges = torch.tensor([[x, y], [y, x]], dtype=torch.long)
		new_edges = torch.tensor([[x], [y]], dtype=torch.long)  # 增加了选择空间  并且可以换一换方向然后再重新试一试 [[y], [x]]

		if self.added_edges == None:
			self.added_edges = new_edges
		else:
			self.added_edges = torch.cat([self.added_edges, new_edges], dim=1)
		# self.new_data.edge_index = torch.cat([self.new_data.edge_index, new_edges], dim=1)
		self.modified = True

	def get_new_edges(self):
		# assert self.modified == True
		return self.added_edges


class NodeAttakEnv(object):
	def __init__(self, data, ctrled_accounts, all_targets,all_labeled_accounts, classifier,args):
		self.classifier = classifier
		self.all_targets = all_targets
		self.static_data = data
		self.agent_ctrled_accounts = defaultdict(dict)
		self.all_ctrled_accounts = defaultdict(list)
		self.edge_nums = self.static_data.edge_index.shape[1]
		self.args = args
		for t in all_labeled_accounts:
			for accounts in ctrled_accounts.values():
				self.all_ctrled_accounts[t] += deepcopy(accounts)
		for a_id in ctrled_accounts:
			agent_ctrled_accounts = defaultdict(list)
			for t in all_labeled_accounts:
				agent_ctrled_accounts[t] = deepcopy(ctrled_accounts[a_id])
			self.agent_ctrled_accounts[a_id] = agent_ctrled_accounts
		self.encoder = GCN(nfeat=data.x.size(1), nhid=args.latent_dim, dropout=0,
						   nlayers=args.layer_num, nclass=max(data.y).item() + 1, with_bn=args.with_bn,
						   weight_decay=args.weight_decay,
						   device=args.device)
		self.encoder = self.encoder.to(args.device)
		self.encoder.fit(data, train_iters=args.epochs, patience=1000, verbose=True)
		with torch.no_grad():
			self.state_origin = self.encoder.layers[0](data.x,data.edge_index)

	def setup(self, target_nodes):  # 重置所有的目标节点，初始节点，和记录修改边的图结构。
		self.target_nodes = target_nodes
		self.n_steps = 0
		self.rewards = None
		self.binary_rewards = None
		self.modified_list = []
		self.list_action_space = deepcopy(self.agent_ctrled_accounts)

		for i in range(len(self.target_nodes)):
			self.modified_list.append(ModifiedGraph())

		self.list_acc_of_all = []

	def step(self, actions,first_nodes=None, start=None,eval_flag=False,reverse=False):

		for i,target_node in enumerate(first_nodes):
			if reverse:
				self.modified_list[start+i].add_edge( actions[i],target_node)
			else:
				self.modified_list[start+i].add_edge(target_node,actions[i])
			self.n_steps += 1

		self.banned_list = None
		acc_list = []
		loss_list = []
		temp_new_data = deepcopy(self.static_data).to("cpu")

			# for i in tqdm(range(len(self.target_nodes))):

		for i in range(len(self.target_nodes)):
			new_edges = self.modified_list[i].get_new_edges()
			temp_new_data.edge_index = torch.cat([temp_new_data.edge_index, new_edges], dim=1)

		pred_log, label_log, prob_log, correct, loss = eval(
				self.classifier, temp_new_data.to(self.args.device), self.target_nodes,self.args)

		# if eval_flag:
				# print(f"new prob: {prob_log[0]:.4f}")
		print(f"correct nums: {sum(correct)}, the prediction accuracy: {sum(correct)/len(self.target_nodes):0.4f}")
			# cur_idx = self.all_targets.index(self.target_nodes[i])
		self.list_acc_of_all.append(np.array(correct))
		self.binary_rewards = (np.array(correct) * -2.0 + 1.0).astype(np.float32)

		if self.args.reward_type == "binary":
				self.rewards = (np.array(correct) * -2.0 + 1.0).astype(np.float32)
		else:
				assert self.args.reward_type == "nll"
				# self.rewards = np.array(loss).astype(np.float32)
				self.rewards = (np.array(correct) * -2.0 + 1.0).astype(np.float32)
				# self.rewards = np.sum(np.array(correct) * -2.0 + 1.0)
			#print(self.rewards)

	def sample_pos_rewards(self, num_samples):
		assert self.list_acc_of_all is not None
		cands = []
		for i in range(len(self.list_acc_of_all)):
			succ = np.where(self.list_acc_of_all[i] < 0.9)[0]
			for j in range(len(succ)):
				cands.append((i, self.all_targets[succ[j]]))
		if num_samples > len(cands):
			return cands
		rd.shuffle(cands)
		return cands[0:num_samples]

	def uniformRandActions(self, a_id,target_list,timestamp=None):
		if timestamp is None:
			act_list = []
			offset = 0
			for i in range(len(self.target_nodes)):
				cur_node = self.target_nodes[i]
				region = self.list_action_space[a_id][cur_node]
				cur_action = region[np.random.randint(len(region))]
				act_list.append(cur_action)
		else:
			act_list = []
			offset = 0
			for i in range(len(target_list)):
				cur_node = target_list[i]
				region = self.list_action_space[a_id][cur_node]
				cur_action = region[np.random.randint(len(region))]
				act_list.append(cur_action)
		return act_list

	def isTerminal(self):
		# if self.n_steps == 2 * self.args.num_mod:
		# 	return True
		# if (self.n_steps) * len(self.target_nodes) >= self.args.ptb_rate * self.edge_nums:
		# 	return True
		if hasattr(self,"test_phase") and self.test_phase is True and self.n_steps == int(self.args.ptb_rate * (self.edge_nums//2)):
			return True
		if self.n_steps == int(self.args.rl_train_ptb_rate * (self.edge_nums//2)):
			return True
		return False

	def getStateRef(self,target_list,timestamp=None):
		if timestamp is None:
			return list(zip(self.target_nodes, self.modified_list))
		else:
			return list(zip(target_list, self.modified_list[timestamp[0]%len(self.target_nodes):timestamp[-1]%len(self.target_nodes)+1])) #,self.state_origin

	def cloneState(self,target_list,timestamp=None):
		if timestamp is None:
			return list(zip(self.target_nodes[:], deepcopy(self.modified_list)))
		else:
			# return list(zip())
			return list(zip(target_list[:],deepcopy(self.modified_list[timestamp[0]%len(self.target_nodes):timestamp[-1]%len(self.target_nodes)+1]))) #, deepcopy(self.state_origin)