import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
import torch_scatter

class InteractionNetwork(pyg.nn.MessagePassing):
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp_edge = nn.Sequential(
            nn.Linear(3*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.mlp_node = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x, edge_index, edge_feature):
        edge_out, aggr = self.propagate(edge_index, x=(x, x), edge_feature=edge_feature)
        node_out = self.mlp_node(torch.cat((x, aggr), dim=-1))
        edge_out = edge_feature + edge_out
        node_out = x + node_out
        return node_out, edge_out

    def message(self, x_i, x_j, edge_feature):
        x = torch.cat([x_i, x_j, edge_feature], dim=-1)
        x = self.mlp_edge(x)
        return x
    
    def aggregate(self, inputs, index, dim_size=None):
        out = torch_scatter.scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce='mean')
        return inputs, out
    
    
class Model(nn.Module):
    def __init__(self, hidden_dim=128, n_mp_layers=6):
        super().__init__()
        
        self.hidden_dim = hidden_dim#
        self.n_mp_layers = n_mp_layers
        self.encode = nn.Linear(3, hidden_dim)
        self.decode = nn.Linear(hidden_dim, 3)
        self.layers = nn.ModuleList([InteractionNetwork(hidden_dim) for _ in range(n_mp_layers)])
        
    def forward(self, data):
        node_feature = torch.zeros((len(data.pos), self.hidden_dim), device=data.pos.device)#
        edge_feature = self.encode(data.edge_attr)
        for i in range(self.n_mp_layers):
            node_feature, edge_feature = self.layers[i](node_feature, data.edge_index, edge_feature=edge_feature)
        out = F.normalize(self.decode(node_feature))
        return out