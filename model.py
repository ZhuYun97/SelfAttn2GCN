from torch_geometric.nn import GCNConv, BatchNorm, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn.inits import glorot
import torch.nn.functional as F
import torch


class GCN(torch.nn.Module):
    def __init__(self, input_dim=768, layer_num=2, hidden=128, class_num=2, activation="relu", pooling="first"):
        super(GCN, self).__init__()
        self.layer_num = layer_num
        self.hidden = hidden
        self.input_dim = input_dim
        self.activation = self.get_activation(activation)
        self.pooling_type = pooling
        self.pooling = self.get_pooling(pooling)
        self.convs = torch.nn.ModuleList()
        self.classifier = GCNConv(hidden, class_num)
        self.bns = torch.nn.ModuleList()
        if self.layer_num >= 1:
            self.convs.append(GCNConv(input_dim, hidden))
            glorot(self.convs[0].weight)
            for i in range(layer_num-1):
                self.convs.append(GCNConv(hidden, hidden))
                glorot(self.convs[i].weight) # initialization
            self.convs.append(GCNConv(hidden, hidden))
            for i in range(layer_num):
                self.bns.append(BatchNorm(hidden))
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(self.layer_num):
            x = self.activation(self.convs[i](x, edge_index))
            # x = self.activation(self.bns[i](self.convs[i](x, edge_index)))
        x = self.classifier(x, edge_index)
        
        if self.pooling_type == "first":
            x = self.pooling(x, data.ptr)
        else:
            x = self.pooling(x, data.batch)
        return x
    
    def reset_parameters(self):
        for i in range(self.layer_num):
            self.convs[i].reset_parameters()
            self.bns[i].reset_parameters()

    def get_activation(self, name: str):
        activations = {
        'relu': F.relu,
        'hardtanh': F.hardtanh,
        'elu': F.elu,
        'leakyrelu': F.leaky_relu,
        'prelu': torch.nn.PReLU(),
        'rrelu': F.rrelu
    }
        return activations[name]

    def get_pooling(self, name: str):
        poolings = {
            "mean": global_mean_pool,
            "add": global_add_pool,
            "max": global_max_pool,
            "first": first_pool
        }
        return poolings[name]
    
def first_pool(x, ptr):
    ptr =  ptr[:-1]
    return x[ptr]