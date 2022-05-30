from multiprocessing import pool
import torch
import torch_geometric
from torch_geometric.data import Data, Batch
import scipy.sparse as sp
import numpy as np
from torch_geometric.utils import to_undirected
from .model import GCN

# GCN with Virtual node

def adj2edges(adj):
    n,m = adj.shape
    assert n == m
    adj = sp.coo_matrix(adj.cpu().numpy())
    indices = np.vstack((adj.row, adj.col))
    edges = torch.LongTensor(indices) 
    edges = to_undirected(edges)
    # append edges between CLS and other nodes
    node_idx = torch.arange(1, n)
    CLS_idx = torch.zeros_like(node_idx)
    CLS_edges = torch.vstack((node_idx, CLS_idx))
    torch.cat((edges, CLS_edges), dim=1)
    return edges

def parser(x, attention_scores, mask, threshold=0.5, type="union"):
    assert type in ['union', 'intersection', 'concat']
    assert len(mask.shape) == 2 # must be B*N
    batch_size, max_node_num = mask.shape
    batched_edges = (attention_scores-torch.eye(max_node_num).to(attention_scores.device)) > threshold # remove self-loop
    batched_node_num = mask.sum(dim=1)
    
    # if type == 'union':
    #     x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, max_node_num, -1) # B,N,D
    batched_data = []
    for i in range(batch_size):
        multihead_adj = batched_edges[i][:, :batched_node_num[i], :batched_node_num[i]] # multi-head
        if type == 'union':
            union_adj = torch.sum(multihead_adj, dim=0) # N, N
            # CLS token is connected with each node, but we initialize zero at this step
            union_adj[0, :] = 0
            union_adj[:, 0] = 0

            union_adj = (union_adj >= 1)
            union_edges = adj2edges(union_adj)
            data = Data(x=x[i][:batched_node_num[i],:], edge_index=union_edges)
            batched_data.append(data)
        elif type == 'intersection':
            pass
        elif type == 'concat':
            pass
        else:
            raise NotImplementedError(f"not impletement type: {type}!")
        
    return Batch.from_data_list(batched_data)

if __name__ == "__main__":
    x = torch.rand(2,12,13,768//12)
    attention_scores = torch.rand(2,12,13,13)
    mask = torch.tensor([[1,1,1,1,1,1,1,1,1,1,0,0,0],
                       [1,1,1,1,1,1,1,1,0,0,0,0,0]], dtype=torch.bool)
    batch = parser(x, attention_scores, mask)
    model = GCN(768)
    out = model(batch)
    print(out.shape)
