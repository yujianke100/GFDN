import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import Linear
from torch.utils.data import Dataset
from tqdm import trange
import os
import random


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

        return x_bar, z


class LoadDataset(Dataset):
    def __init__(self, data):
        self.x = data.u_x[data.train_u].cpu()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))

def pretrain_ae(model, x, n_clusters, epochs):
    print(model)
    optimizer = Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        x_bar, _ = model(x)
        loss = F.mse_loss(x_bar, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    root_path, _ = os.path.split(os.path.abspath(__file__))
    torch.save(model.state_dict(), root_path+'/model/ae_pre_train.pkl')
    torch.cuda.empty_cache()
    
def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
def pre_train(dataset, n_clusters, n_input, n_z, n_enc, hidden, pre_ae_epoch):
    setup_seed(10)
    model = AE(
        n_enc=n_enc,
        hidden=hidden,
        n_input=n_input,
        n_z=n_z,).cuda()
    
    pretrain_ae(model, dataset.u_x[dataset.train_u], n_clusters, pre_ae_epoch)
