import pyabcore
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from time import time
import torch
import networkx as nx
from collections import Counter
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data.dataset import Dataset

def get_labeled_node(df):

    pos = df[df[:,-1] == 1]
    neg = df[df[:,-1] == 0]

    total_u = list(set(df[:,0].numpy()))
    total_u.sort()
    pos_u = set(pos[:,0].numpy())

    neg_u = set(neg[:,0].numpy()) - pos_u

    pos_u_mask = torch.BoolTensor([(i in pos_u) for i in total_u])
    neg_u_mask = torch.BoolTensor([(i in neg_u) for i in total_u])
    u_l = (torch.zeros(len(total_u))-1).double()
    u_l = torch.where(neg_u_mask, 0., u_l)
    u_l = torch.where(pos_u_mask, 1., u_l)
    
    return u_l

def get_data():
    root_path, _ = os.path.split(os.path.abspath(__file__)) 

    train_data = pd.read_csv(root_path+'/dataset/train.txt', names=['u', 'i', 'l'], delimiter='\t', dtype=int)
    test_data = pd.read_csv(root_path+'/dataset/test.txt', names=['u', 'i','l'], delimiter='\t', dtype=int)

    df = pd.concat((train_data, test_data))

    dataset = Dataset()

    dataset.max_u = max(df['u'])+1
    dataset.max_i = max(df['i'])+1

    df_labels = df[df['l'] != -1]
    
    dataset.all_edge = np.array(df[['u', 'i']], dtype=np.int32)

    dataset.train_edge = torch.LongTensor(np.array(train_data[['u', 'i', 'l']]))

    dataset.test_edge = torch.LongTensor(np.array(test_data[['u', 'i', 'l']]))
    
    dataset.train_u = list(set(train_data['u']))
    dataset.train_u.sort()
    dataset.train_u = torch.LongTensor(dataset.train_u)

    # dataset.test_u = list(set(test_data['u']))
    # dataset.test_u.sort()
    # dataset.test_u = torch.LongTensor(dataset.test_u)

    dataset.u_x = torch.FloatTensor(np.load(root_path+'/dataset/bdt_u_features.npy'))
    dataset.i_x = torch.FloatTensor(np.load(root_path+'/dataset/bdt_i_features.npy'))
    
    return dataset

def get_abcore(dataset, device):
    # print('build index')
    abcore = pyabcore.Pyabcore(dataset.max_u, dataset.max_i)
    # start_time = time()
    abcore.index(dataset.all_edge)
    # index_time = time()
    # print('finished, time:{}'.format(index_time - start_time))
    a = 2
    b = 1
    dataset.core_u_x = torch.BoolTensor([])
    dataset.core_i_x = torch.BoolTensor([])
    while 1:
        abcore.query(a, b)
        result_u = torch.BoolTensor(abcore.get_left())
        result_i = torch.BoolTensor(abcore.get_right())
        if(result_i.sum() < len(result_i)*0.01):
            print('max b:{}'.format(b-1))
            dataset.max_b = b-1
            break

        dataset.core_u_x = torch.cat((dataset.core_u_x, result_u.unsqueeze(-1)),dim=1)
        dataset.core_i_x = torch.cat((dataset.core_i_x, result_i.unsqueeze(-1)),dim=1)
        b += 1

    
    tmp_edge = dataset.test_edge.clone()
    tmp_edge[:,-1] = -1
    dataset.train_u_l = get_labeled_node(torch.cat((dataset.train_edge, tmp_edge))).to(device)

    tmp_edge = dataset.train_edge.clone()
    tmp_edge[:,-1] = -1

    dataset.test_u_l = get_labeled_node(torch.cat((tmp_edge, dataset.test_edge))).to(device)

    tmp_label = torch.where(dataset.test_u_l == -1, -1., dataset.train_u_l)
    dataset.test_u_l = torch.where((tmp_label == 1) * (dataset.test_u_l == 0), 1., dataset.test_u_l)

    dataset.train_u_l = dataset.train_u_l.long().to(device)
    dataset.test_u_l = dataset.test_u_l.long().to(device)

    dataset.test_edge = torch.cat((tmp_edge, dataset.test_edge))
    
    dataset.u_x = torch.cat((dataset.core_u_x, dataset.u_x),dim=1).to(device)
    dataset.i_x = torch.cat((dataset.core_i_x,dataset.i_x),dim=1).to(device)

    dataset.train_edge_x = torch.cat((dataset.u_x[dataset.train_edge[:,0]][:,:dataset.max_b],dataset.i_x[dataset.train_edge[:,1]][:,:dataset.max_b],dataset.u_x[dataset.train_edge[:,0]][:,dataset.max_b:],dataset.i_x[dataset.train_edge[:,1]][:,dataset.max_b:]),dim=1).to(device)

    dataset.test_edge_x = torch.cat((dataset.u_x[dataset.test_edge[:,0]][:,:dataset.max_b],dataset.i_x[dataset.test_edge[:,1]][:,:dataset.max_b],dataset.u_x[dataset.test_edge[:,0]][:,dataset.max_b:],dataset.i_x[dataset.test_edge[:,1]][:,dataset.max_b:]),dim=1).to(device)

    dataset.train_edge_y = dataset.train_edge[:,2].to(device)
    dataset.train_edge = dataset.train_edge[:,:2].t().to(device)
    dataset.train_edge[1] = dataset.train_edge[1] + dataset.max_u

    dataset.test_edge_y = dataset.test_edge[:,2].to(device)
    dataset.test_edge = dataset.test_edge[:,:2].t().to(device)
    dataset.test_edge[1] = dataset.test_edge[1] + dataset.max_u

    return dataset


def get_abcore_data(device):
    dataset= get_data()
    dataset = get_abcore(dataset, device)

    return dataset
    
if __name__ == '__main__':
    get_abcore_data('cpu')