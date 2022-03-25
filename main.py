from __future__ import print_function, division
import argparse
import random
import numpy as np
# from sklearn.cluster import KMeans
from kmeans import kmeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.nn import Linear
from GNN import *
from abcore_data import get_abcore_data
from pre_train import pre_train
import os
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from focal_loss import FocalLoss
from torch_geometric.loader import NeighborSampler
from tqdm import tqdm

class AE(nn.Module):

    def __init__(self, n_enc, hidden,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_in = Linear(n_input, n_enc)
        self.hidden_enc = nn.ModuleList([Linear(n_enc, n_enc) for i in range(hidden)])
        self.z_layer = Linear(n_enc, n_z)

        self.dec_in = Linear(n_z, n_enc)
        self.hidden_dec = nn.ModuleList([Linear(n_enc, n_enc) for i in range(hidden)])
        self.x_bar_layer = Linear(n_enc, n_input)

    def forward(self, x):
        enc_result = []
        enc_result.append(F.relu(self.enc_in(x)))
        for layer in self.hidden_enc:
            enc_result.append(F.relu(layer(enc_result[-1])))
        z = self.z_layer(enc_result[-1])

        dec = F.relu(self.dec_in(z))
        for layer in self.hidden_dec:
            dec = F.relu(layer(dec))
        x_bar = self.x_bar_layer(dec)

        return x_bar, enc_result , z

class CORE(nn.Module):
    def __init__(self, max_b):
        super(CORE, self).__init__()
        self.weight = Parameter(torch.ones([max_b]))

    def forward(self, x):
        return F.softmax(self.weight * x, dim=1)

class TL(nn.Module):
    def __init__(self, n_clusters):
        super(TL, self).__init__()
        self.to_label1 = Linear(n_clusters, n_clusters//2)
        self.to_label2 = Linear(n_clusters//2, 2)

    def forward(self, x):
        x = F.relu(self.to_label1(x))
        x = F.softmax(self.to_label2(x), dim=1)
        return x

class TEL(nn.Module):
    def __init__(self, n):
        super(TEL, self).__init__()
        self.to_edge_label1 = Linear(n, 64)
        self.to_edge_label2 = Linear(64, 2)

    def forward(self, edge_x):
        
        x = F.relu(self.to_edge_label1(edge_x))
        x = F.softmax(self.to_edge_label2(x), dim=1)
        return x

class GNN(nn.Module):
    def __init__(self, n_input, n_i, n_clusters, n_enc, hidden, n_z, pre_ae_epoch, max_b):
        super(GNN, self).__init__()
        if(args.gnn == 'sage'):
            GNN_NET = SAGE_NET
        else:
            GNN_NET = GCN_NET
        self.core_u = CORE(max_b)
        self.core_i = CORE(max_b)
        self.n_input = n_input
        self.ae = AE(n_enc, hidden,
                 n_input, n_z)
        self.tl = TL(n_clusters = n_clusters)
        self.tel = TEL(1 + n_clusters + n_input + n_i)
        self.i_emb = Linear(n_i-max_b, n_input-max_b)
        self.gnn_in = GNN_NET(n_input, n_enc)
        self.hidden_gnn = nn.ModuleList([GNN_NET(n_enc, n_enc) for i in range(hidden)])
        self.gnn_nz = GNN_NET(n_enc, n_z)
        self.gnn_cluster = GNN_NET(n_z, n_clusters)

        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = 1
        if(not os.path.exists(root_path + '/model/model_{}.pkl'.format(args.epoch))):
            pre_train(dataset, n_clusters, n_input, n_z, n_enc, hidden, pre_ae_epoch)
        self.ae.load_state_dict(torch.load(root_path+'/model/ae_pre_train.pkl', map_location='cpu'))

    def forward(self, x, edge_u_x, edge_u_id, edge_index, train=True):
        q = 0

        x_bar, h, z = self.ae(edge_u_x)
        x = self.gnn_in(x, edge_index)
        for i, layer in enumerate(self.hidden_gnn):
            x[edge_u_id] = x[edge_u_id] + h[i]
            x = layer(x, edge_index)
        x[edge_u_id] = x[edge_u_id] + h[-1]
        x = self.gnn_nz(x, edge_index)
        x[edge_u_id] = x[edge_u_id] + z
        x = self.gnn_cluster(x, edge_index, active=False)

        x = torch.sigmoid(x[edge_u_id])

        if(train):
            q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
            q = q.pow((self.v + 1.0) / 2.0)
            q = (q.t() / torch.sum(q, 1)).t()

        return x, x_bar, q

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def init_data(adjs, n_id, train=True):
    u_id_mask = (n_id < dataset.max_u)
    u_id = n_id[u_id_mask]

    edge_u_id = list(set(adjs.edge_index[0].numpy()))
    edge_u_id.sort()
    edge_u_id = torch.LongTensor(edge_u_id)

    i_id_mask = (n_id >= dataset.max_u)
    i_id = n_id[i_id_mask] - dataset.max_u

    u_x = dataset.u_x[u_id]
    loss_u_x = dataset.u_x[n_id[edge_u_id]]
    i_x = dataset.i_x[i_id]

    if(train):
        u_l = dataset.train_u_l[n_id[edge_u_id]]
        u_l_mask = (u_l != -1)
        u_l_y = u_l[u_l_mask]

        edge_y = dataset.train_edge_y[adjs.e_id]
        edge_y_mask = (edge_y != -1)
        edge_y = edge_y[edge_y_mask]
        edge_x = dataset.train_edge_x[adjs.e_id][edge_y_mask]
        
    else:
        u_l = dataset.test_u_l[n_id[edge_u_id]]
        u_l_mask = (u_l != -1)
        u_l_y = u_l[u_l_mask]

        edge_y = dataset.test_edge_y[adjs.e_id]
        edge_y_mask = (edge_y != -1)
        edge_y = edge_y[edge_y_mask]
        edge_x = dataset.test_edge_x[adjs.e_id][edge_y_mask]
    
    return [u_x[:,:dataset.max_b],u_x[:,dataset.max_b:], i_x[:,:dataset.max_b], i_x[:,dataset.max_b:], u_id_mask, i_id_mask, edge_u_id, u_l_mask, edge_y_mask, u_l_y, edge_x[:,:dataset.max_b],edge_x[:,dataset.max_b:dataset.max_b*2], edge_x[:,dataset.max_b*2:], edge_y, loss_u_x]

def train(adjs, n_id, train=True, test_init_data=None, final_epoch = False):
    global model
    global optimizer
    global dataset
    global focal_loss
    if(train):
        model.train()
        optimizer.zero_grad()
        u_x_core, u_x, i_x_core, i_x, u_id_mask, i_id_mask, edge_u_id, u_l_mask, edge_y_mask, u_l_y, edge_x_u_core, edge_x_i_core, edge_x_out_core, edge_y, loss_u_x = init_data(adjs, n_id, train)
        adjs = adjs.to(device)
    else:
        torch.cuda.empty_cache()
        model.eval()
        u_x_core, u_x, i_x_core, i_x, u_id_mask, i_id_mask, edge_u_id, u_l_mask, edge_y_mask, u_l_y, edge_x_u_core, edge_x_i_core, edge_x_out_core, edge_y, loss_u_x = test_init_data

    x = torch.zeros([len(n_id), u_x.shape[1]+dataset.max_b]).to(device)

    x[u_id_mask] = torch.cat((model.core_u(u_x_core), u_x),dim=1)
    x[i_id_mask] = torch.cat((model.core_i(i_x_core),model.i_emb(i_x)),dim=1)
    edge_x = torch.cat((model.core_u(edge_x_u_core), model.core_i(edge_x_i_core), edge_x_out_core),dim=1)
    edge_u_x = x[edge_u_id]

    x, x_bar, q = model(x, edge_u_x, edge_u_id, adjs.edge_index)
    p = 0
    if(train):
        p = target_distribution(q)

    pre_l = torch.zeros([len(n_id), 2]).to(device)
    tmp_pre_l = torch.zeros([len(x), 2]).to(device)
    loss_pre_l = model.tl(x[u_l_mask])
    tmp_pre_l[u_l_mask] = loss_pre_l
    pre_l[edge_u_id] = tmp_pre_l

    total_x = torch.zeros([len(n_id), x.shape[1]]).to(device)
    total_x[edge_u_id] = x
    
    edge_us = adjs.edge_index[0][edge_y_mask]
    pre_e_l = model.tel(torch.cat((pre_l[:,1][edge_us].unsqueeze(-1), total_x[edge_us] ,edge_x),dim=1))
    
    if(train):
        label_loss = focal_loss(loss_pre_l, u_l_y)
        edge_loss = focal_loss(pre_e_l, edge_y)
        ae_loss = F.mse_loss(x_bar, loss_u_x)
        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')

        loss = args.ll * label_loss + args.el * edge_loss + args.al * ae_loss + args.kl * kl_loss
        loss.backward()
        optimizer.step()
    edge_y = edge_y.cpu()
    pre_e_l = pre_e_l[:,1].cpu().detach()
    
    max_th = args.th
    auc = roc_auc_score(edge_y, pre_e_l)
    
    pre_result = (pre_e_l > args.th)
    f1 = f1_score(edge_y, pre_result)

    if(train):
        return loss.item(), f1, auc
    else:
        pre = precision_score(edge_y, pre_result)
        acc = accuracy_score(edge_y, pre_result)
        recall = recall_score(edge_y, pre_result)
        if(final_epoch):
            torch.save(x, root_path+'/model/group_pred.pkl')
            torch.save(n_id[edge_u_id], root_path+'/model/group_pred_uid.pkl')
            torch.save(model.core_u.weight, root_path+'/model/core_u_weight.pkl')
            torch.save(model.core_i.weight, root_path+'/model/core_i_weight.pkl')
        torch.cuda.empty_cache()
        return auc, f1, acc, pre, recall, max_th

def train_exp(dataset):
    global model
    global optimizer
    model = GNN(
                n_input=dataset.u_x.shape[1],
                n_i=dataset.i_x.shape[1],
                n_clusters=args.n_clusters,
                n_enc = args.hidden_dim, 
                hidden = args.hidden,
                n_z=args.n_z,
                pre_ae_epoch = args.pre_ae_epoch,
                max_b = dataset.max_b
                ).to(device)
    print(model)
    print(args)

    optimizer = Adam(model.parameters(), lr=args.lr)
    # KNN Graph
    
    train_loader = NeighborSampler(dataset.train_edge, sizes=[-1], batch_size=9000, num_workers=6, shuffle=True)
    test_loader = NeighborSampler(dataset.test_edge, sizes=[-1], batch_size=99999999, num_workers=6, shuffle=False)
    assert len(test_loader) == 1

    batch_num = len(train_loader)
    with torch.no_grad():
        _, _, z = model.ae(dataset.u_x)
    model.cluster_layer.data = kmeans(z.data, args.n_clusters).to(device)
    torch.cuda.empty_cache()
    max_train_f1 = 0
    max_test_f1 = 0
    max_test_acc = 0
    max_test_pre = 0
    max_test_recall = 0
    max_test_auc = 0
    max_epoch = 0

    for batch_size, n_id, adjs in test_loader:
        test_init_data = init_data(adjs, n_id, train=False)
        test_adjs = adjs.to(device)
        test_n_id = n_id
    for epoch in range(1, args.epoch+1):
        total_loss = total_f1 = total_auc = 0
        for batch_size, n_id, adjs in tqdm(train_loader):
            loss, f1, auc = train(adjs, n_id)
            total_loss += loss
            total_f1 += f1
            total_auc += auc
        loss = total_loss / batch_num
        f1 = total_f1 / batch_num
        auc = total_auc / batch_num
        
        test_auc, test_f1, test_acc, test_pre, test_recall, max_th = train(test_adjs, test_n_id, train=False, test_init_data=test_init_data, final_epoch=(epoch==args.epoch))
        print('{} loss:{:.5f} f1:{:.4f} tauc:{:.4f} tf1:{:.4f} tacc:{:.4f} tpre:{:.4f} trecall:{:.4f} th:{}'.format(epoch, loss, f1, test_auc, test_f1, test_acc, test_pre, test_recall, max_th))

        if(f1 > max_train_f1):
            max_train_f1 = f1
            max_test_f1 = test_f1
            max_test_auc = test_auc
            max_test_acc = test_acc
            max_test_pre = test_pre
            max_test_recall = test_recall
            max_epoch = epoch

    print('max epoch:{} auc:{:.4f} f1:{:.4f} acc:{:.4f} pre:{:.4f} recall:{:.4f}'.format(max_epoch,max_test_auc, max_test_f1, max_test_acc, max_test_pre, max_test_recall))

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_clusters', default=32, type=int)
    parser.add_argument('--n_z', default=64, type=int)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--pre_ae_epoch', default=150, type=int)
    parser.add_argument('--seed', default=10, type=int)
    parser.add_argument('--hidden', default=1, type=int)

    parser.add_argument('--ll', type=float, default=0.1)
    parser.add_argument('--el', type=float, default=0.6)
    parser.add_argument('--al', type=float, default=0.2)
    parser.add_argument('--kl', type=float, default=0.1)

    parser.add_argument('--th', type=float, default=0.45)
    parser.add_argument('--gnn', type=str, default='sage')
    
    args = parser.parse_args()
    setup_seed(args.seed)
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")
    focal_loss = FocalLoss(2)
    dataset = get_abcore_data(device)
    root_path, _ = os.path.split(os.path.abspath(__file__)) 

    if(os.path.isdir(root_path + '/model') == False):
	    os.mkdir(root_path + '/model')

    train_exp(dataset)