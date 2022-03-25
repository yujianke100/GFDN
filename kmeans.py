import torch
def kmeans(x, ncluster, niter=10):
    '''
    x : torch.tensor(data_num,data_dim)
    ncluster : The number of clustering for data_num
    niter : Number of iterations for kmeans
    '''
    N, D = x.size()
    c = x[torch.randperm(N)[:ncluster]] # init clusters at random
    for i in range(niter):
        a = ((x[:, None, :] - c[None, :, :])**2).sum(-1).argmin(1)
        c = torch.stack([x[a==k].mean(0) for k in range(ncluster)])
        nanix = torch.any(torch.isnan(c), dim=1)
        ndead = nanix.sum().item()
        c[nanix] = x[torch.randperm(N)[:ndead]]
    return c