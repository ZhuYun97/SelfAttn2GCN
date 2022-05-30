from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv, BatchNorm, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn.inits import glorot
import torch.nn.functional as F
import torch


class GCN(torch.nn.Module):
    def __init__(self, input_dim=768, layer_num=1, hidden=768, class_num=2, activation="relu", pooling="first", gnn_type="sage"):
        super(GCN, self).__init__()
        self.layer_num = layer_num
        self.hidden = hidden
        self.input_dim = input_dim
        self.activation = self.get_activation(activation)
        self.pooling_type = pooling
        self.pooling = self.get_pooling(pooling)
        self.convs = torch.nn.ModuleList()
        gnn_conv = self.get_gnn_conv(gnn_type)
        self.classifier = gnn_conv(hidden, class_num)
        self.bns = torch.nn.ModuleList()
        if self.layer_num >= 1:
            self.convs.append(gnn_conv(input_dim, hidden))
            # glorot(self.convs[0].weight)
            for i in range(layer_num-1):
                self.convs.append(gnn_conv(hidden, hidden))
                # glorot(self.convs[i].weight) # initialization
            self.convs.append(gnn_conv(hidden, hidden))
            for i in range(layer_num):
                self.bns.append(torch.nn.BatchNorm1d(hidden))
        ### List of MLPs to transform virtual node at every layer
        # self.mlp_virtualnode_list = torch.nn.ModuleList()
    
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
    def get_gnn_conv(self, name: str):
        gnns = {
            "gcn": GCNConv,
            "sage": SAGEConv,
            "gin": GINConv,
            "gat": GATConv
        }
        return gnns[name]
    
def first_pool(x, ptr):
    ptr =  ptr[:-1]
    return x[ptr]
