import torch
from deeprobust.graph.defense import ProGNN
from deeprobust.graph.defense_pyg import GCN,AirGNN,SAGE,GAT,APPNP,GPRGNN,SGC,ARMA
from deeprobust.graph.utils import get_perf
import os
# def GCN_defense(features, adj ,labels, idx_train, idx_test,nhid,device,epochs,verbose):
#     # Setup GCN Model
#     model = GCN(nfeat=features.shape[1], nhid=nhid, nclass=(labels.max() + 1).item(), device=device)
#     model = model.to(device)
#     features.to(device)
#     adj.to(device)
#     labels.to(device)
#     model.fit(features,adj,labels, idx_train, train_iters=epochs,verbose=verbose)
#     model.eval()
#     model.test(idx_test)
#     features.cpu()
#     adj.cpu()
#     labels.cpu()


def GCN_defense(data,nhid,device,epochs,args,attacked=False):
    feat, labels = data.x, data.y
    args.defense_path = os.path.join(args.base_path, args.dataset + f"_{args.seed}_" + args.defense_algorithm+".model")
    if os.path.exists(args.defense_path):
        model = torch.load(args.defense_path,map_location=args.map_location)
    else:
        model = GCN(nfeat=feat.shape[1], nhid=args.hid_dim, dropout=0,
                nlayers=args.layer_num, nclass=max(labels).item() + 1,with_bn=args.with_bn,weight_decay=args.weight_decay,
                device=device)
        torch.save(model,args.defense_path)
    args.defense_trained_path = os.path.join(args.base_path,
                                     args.dataset + f"_{args.seed}_" + args.defense_algorithm + "_trained.model")
    if attacked:
        model = torch.load(args.defense_trained_path,map_location=args.map_location)
        model.device = device
        model = model.to(device)
    else:
        model.device = device
        model = model.to(device)
        model.fit(data, train_iters=epochs, patience=1000, verbose=True)
    model.eval()
    model.data = data.to(device)
    output = model.predict()
    labels = labels.to(device)
    if not attacked:
        torch.save(model, args.defense_trained_path)
    print("Test set results:", get_perf(output, labels, data.test_mask, verbose=0)[1])
    return get_perf(output, labels, data.test_mask, verbose=0)[1]

def SGC_defense(data,nhid,device,epochs,args,attacked=False):
    feat, labels = data.x, data.y
    args.defense_path = os.path.join(args.base_path,
                                     args.dataset + f"_{args.seed}_" + args.defense_algorithm + ".model")
    if os.path.exists(args.defense_path):
        model = torch.load(args.defense_path, map_location=args.map_location)
    else:
        model = SGC(nfeat=feat.shape[1], nhid=args.hid_dim, dropout=0,
                    nlayers=args.layer_num, nclass=max(labels).item() + 1, with_bn=args.with_bn,
                    weight_decay=args.weight_decay,
                    device=device)
        torch.save(model, args.defense_path)
    args.defense_trained_path = os.path.join(args.base_path,
                                             args.dataset + f"_{args.seed}_" + args.defense_algorithm + "_trained.model")
    if attacked:
        model = torch.load(args.defense_trained_path, map_location=args.map_location)
        model.device = device
        model = model.to(device)
    else:
        model.device = device
        model = model.to(device)
        model.fit(data, train_iters=epochs, patience=1000, verbose=True)
    model.eval()
    model.data = data.to(device)
    output = model.predict()
    labels = labels.to(device)
    if not attacked:
        torch.save(model, args.defense_trained_path)
    print("Test set results:", get_perf(output, labels, data.test_mask, verbose=0)[1])

def ARMA_defense(data,nhid,device,epochs,args,attacked=False):
    feat, labels = data.x, data.y
    args.defense_path = os.path.join(args.base_path,
                                     args.dataset + f"_{args.seed}_" + args.defense_algorithm + ".model")
    if os.path.exists(args.defense_path):
        model = torch.load(args.defense_path, map_location=args.map_location)
    else:
        model = ARMA(nfeat=feat.shape[1], nhid=args.hid_dim, dropout=0,
                    nlayers=args.layer_num, nclass=max(labels).item() + 1, with_bn=args.with_bn,
                    weight_decay=args.weight_decay,
                    device=device)
        torch.save(model, args.defense_path)
    args.defense_trained_path = os.path.join(args.base_path,
                                             args.dataset + f"_{args.seed}_" + args.defense_algorithm + "_trained.model")
    if attacked:
        model = torch.load(args.defense_trained_path, map_location=args.map_location)
        model.device = device
        model = model.to(device)
    else:
        model.device = device
        model = model.to(device)
        model.fit(data, train_iters=epochs, patience=1000, verbose=True)
    model.eval()
    model.data = data.to(device)
    output = model.predict()
    labels = labels.to(device)
    if not attacked:
        torch.save(model, args.defense_trained_path)
    print("Test set results:", get_perf(output, labels, data.test_mask, verbose=0)[1])
def GAT_defense(data,nhid,device,epochs,args,attacked=False):
    feat, labels = data.x, data.y
    args.defense_path = os.path.join(args.base_path,
                                     args.dataset + f"_{args.seed}_" + args.defense_algorithm + ".model")
    if os.path.exists(args.defense_path):
        model = torch.load(args.defense_path,map_location=args.map_location)
    else:
        model = GAT(nfeat=feat.shape[1], nhid=args.hid_dim, heads=8, nlayers=args.layer_num,
              nclass=max(labels).item() + 1, with_bn=args.with_bn, weight_decay=args.weight_decay,
              dropout=0, device=device).to(device)
        torch.save(model, args.defense_path)
    args.defense_trained_path = os.path.join(args.base_path,
                                             args.dataset + f"_{args.seed}_" + args.defense_algorithm + "_trained.model")
    if attacked:
        model = torch.load(args.defense_trained_path, map_location=args.map_location)
        model.device = device
        model = model.to(device)
    else:
        model.device = device
        model = model.to(device)
        model.fit(data, train_iters=epochs, patience=1000, verbose=True)
    model.eval()
    model.data = data.to(device)
    output = model.predict()
    labels = labels.to(device)
    if not attacked:
        torch.save(model, args.defense_trained_path)
    print("Test set results:", get_perf(output, labels, data.test_mask, verbose=0)[1])

def ProGNN_defense(features, adj ,labels, idx_train,idx_val ,idx_test,nhid,device,epochs,verbose,args,attacked=False):
    features.to(device)
    adj.to(device)
    labels.to(device)
    args.defense_path = os.path.join(args.base_path,
                                     args.dataset + f"_{args.seed}_" + args.defense_algorithm + ".model")
    if os.path.exists(args.defense_path):
        pro_model = torch.load(args.defense_path,map_location=args.map_location)
    else:
        model = GCN(nfeat=features.shape[1], nhid=nhid, nclass=(labels.max() + 1).item(), device=device)
        pro_model = ProGNN(model, args, device)
        torch.save(pro_model, args.defense_path)
    # model.device = device
    # pro_model = pro_model.to(device)
    pro_model.fit(features,adj,labels,idx_train,idx_val)
    pro_model.test(features,labels,idx_test)

def GPRGNN_defense(data,nhid,device,epochs,args,attacked=False):
    feat, labels = data.x, data.y
    args.defense_path = os.path.join(args.base_path,
                                     args.dataset + f"_{args.seed}_" + args.defense_algorithm + ".model")
    if os.path.exists(args.defense_path):
        model = torch.load(args.defense_path, map_location=args.map_location)
    else:
        #(self, in_channels, hidden_channels, out_channels, Init='PPR', dprate=.5, dropout=.5,lr=0.01, weight_decay=0, device='cpu',K=10, alpha=.1, Gamma=None, ppnp='GPR_prop'):
        model = GPRGNN(in_channels=feat.shape[1], hidden_channels=nhid,out_channels=max(labels).item() + 1, dropout=0,
                       K=10, weight_decay=args.weight_decay, device=device).to(device)
        torch.save(model, args.defense_path)
    # print(model)
    args.defense_trained_path = os.path.join(args.base_path,
                                             args.dataset + f"_{args.seed}_" + args.defense_algorithm + "_trained.model")
    if attacked:
        model = torch.load(args.defense_trained_path, map_location=args.map_location)
        model.device = device
        model = model.to(device)
    else:
        model.device = device
        model = model.to(device)
        model.fit(data, train_iters=epochs, patience=1000, verbose=True)
    model.eval()
    model.data = data.to(device)
    output = model.predict()
    labels = labels.to(device)
    if not attacked:
        torch.save(model, args.defense_trained_path)
    print("Test set results:", get_perf(output, labels, data.test_mask, verbose=0)[1])

def AirGNN_defense(data,nhid,device,epochs,args,attacked=False):
    feat, labels = data.x, data.y
    args.defense_path = os.path.join(args.base_path,
                                     args.dataset + f"_{args.seed}_" + args.defense_algorithm + ".model")
    if os.path.exists(args.defense_path):
        model = torch.load(args.defense_path,map_location=args.map_location)
    else:
        model = AirGNN(nfeat=feat.shape[1], nhid=nhid, dropout=0, with_bn=args.with_bn,
                   K=10, weight_decay=args.weight_decay, args=args, nlayers=args.layer_num,
                   nclass=max(labels).item() + 1, device=device).to(device)
        torch.save(model, args.defense_path)
    # print(model)
    args.defense_trained_path = os.path.join(args.base_path,
                                             args.dataset + f"_{args.seed}_" + args.defense_algorithm + "_trained.model")
    if attacked:
        model = torch.load(args.defense_trained_path, map_location=args.map_location)
        model.device = device
        model = model.to(device)
    else:
        model.device = device
        model = model.to(device)
        model.fit(data, train_iters=epochs, patience=1000, verbose=True)
    model.eval()
    model.data = data.to(device)
    output = model.predict()
    labels = labels.to(device)
    if not attacked:
        torch.save(model, args.defense_trained_path)
    print("Test set results:", get_perf(output, labels, data.test_mask, verbose=0)[1])

def SAGE_defense(data,nhid,device,epochs,args,attacked=False):
    feat, labels = data.x, data.y
    args.defense_path = os.path.join(args.base_path,
                                     args.dataset + f"_{args.seed}_" + args.defense_algorithm + ".model")
    if os.path.exists(args.defense_path):
        model = torch.load(args.defense_path,map_location=args.map_location)
    else:
        model = SAGE(feat.shape[1], args.hid_dim, max(labels).item() + 1, num_layers=5,
                     dropout=0,with_bn=args.with_bn, weight_decay=args.weight_decay, device=device).to(device)
        torch.save(model, args.defense_path)
    # print(model)
    args.defense_trained_path = os.path.join(args.base_path,
                                             args.dataset + f"_{args.seed}_" + args.defense_algorithm + "_trained.model")
    if attacked:
        model = torch.load(args.defense_trained_path, map_location=args.map_location)
        model.device = device
        model = model.to(device)
    else:
        model.device = device
        model = model.to(device)
        model.fit(data, train_iters=epochs, patience=1000, verbose=True)
    model.eval()
    model.data = data.to(device)
    output = model.predict()
    labels = labels.to(device)
    if not attacked:
        torch.save(model, args.defense_trained_path)
    print("Test set results:", get_perf(output, labels, data.test_mask, verbose=0)[1])



def APPNP_defense(data,nhid,device,epochs,args,attacked=False):
    feat, labels = data.x, data.y
    args.defense_path = os.path.join(args.base_path,
                                     args.dataset + f"_{args.seed}_" + args.defense_algorithm + ".model")
    if os.path.exists(args.defense_path):
        model = torch.load(args.defense_path,map_location=args.map_location)
    else:
        model = APPNP(nfeat=feat.shape[1], nhid=args.hid_dim,K=10,alpha=0.1, dropout=0,
                nclass=max(labels).item() + 1, with_bn=args.with_bn, weight_decay=args.weight_decay,device=device).to(device)
        torch.save(model, args.defense_path)
    args.defense_trained_path = os.path.join(args.base_path,
                                             args.dataset + f"_{args.seed}_" + args.defense_algorithm + "_trained.model")
    if attacked:
        model = torch.load(args.defense_trained_path, map_location=args.map_location)
        model.device = device
        model = model.to(device)
    else:
        model.device = device
        model = model.to(device)
        model.fit(data, train_iters=epochs, patience=1000, verbose=True)
    model.eval()
    model.data = data.to(device)
    output = model.predict()
    labels = labels.to(device)
    if not attacked:
        torch.save(model, args.defense_trained_path)
    print("Test set results:", get_perf(output, labels, data.test_mask, verbose=0)[1])
