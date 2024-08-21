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

class SimpleNet_TimeDependent(nn.Module):
    def __init__(self, init_eval):
        super(SimpleNet_TimeDependent, self).__init__()
        self.init_eval = init_eval
        self.fc0 = nn.Linear(4, 128)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        
        nn.init.xavier_uniform_(self.fc0.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        
    def forward(self, t, x):
        tx = torch.concat((t, x),1)
        u = torch.sin(torch.pi*self.fc0(tx))
        u = torch.sin(torch.pi*self.fc1(u))
        u = torch.sin(torch.pi*self.fc2(u))
        u = self.fc3(u)
        #return self.init_eval(x) + (1-torch.exp(t))*u
        return self.init_eval(x) + t*u
    
# class SimpleNet_TimeIndependent(nn.Module):
#     def __init__(self):
#         super(SimpleNet_TimeIndependent, self).__init__()
#         self.fc0 = nn.Linear(3, 128)
#         self.fc1 = nn.Linear(128, 128)
#         self.fc2 = nn.Linear(128, 128)
#         self.fc3 = nn.Linear(128, 1)
        
#         nn.init.xavier_normal_(self.fc0.weight)
#         nn.init.xavier_normal_(self.fc1.weight)
#         nn.init.xavier_normal_(self.fc2.weight)
#         nn.init.xavier_normal_(self.fc3.weight)
        
#     def forward(self, x):
#         u = torch.sin(torch.pi*self.fc0(x))
#         u = torch.sin(torch.pi*self.fc1(u))
#         u = torch.sin(torch.pi*self.fc2(u))
#         u = self.fc3(u)
#         return u
