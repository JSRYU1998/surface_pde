import torch
import torch.nn.functional as F
import torch_geometric as pyg
import numpy as np

#Define some auxiliary functions
def expand(ind, N): # ind : 1d torch array
    ind = ind.reshape(-1,1)
    ind = ind.repeat(1,N)
    ind = ind.reshape(-1)
    return ind

#Define graph dataset for test
#position_path : X (position_vectors), which is of shape (num_pts, 3)
#normal_path : y_true (normal_vectors) or None (if we don't know the target), which is of shape (num_pts, 3)
class GraphDataset(pyg.data.Dataset):
    def __init__(self, position_path, normal_path, K_Main, Is_Normalize):
        super().__init__()
        self.K_Main = K_Main
        self.Is_Normalize = Is_Normalize
        assert type(Is_Normalize) == bool
        self.X = torch.from_numpy(np.load(position_path)).float().unsqueeze(dim=0) #(num_pts, 3) --> (1, num_pts, 3)
        if normal_path == None:  #(if we don't know the target)
            self.Y = None
        else: #(if we know the target)
            self.Y = torch.from_numpy(np.load(normal_path)).float().unsqueeze(dim=0) #(num_pts, 3) --> (1, num_pts, 3)

    def len(self):
        return len(self.X)
    
    def get(self, idx):
        node_loc = self.X[idx]
        
        # construct the graph using the KNN
        n_node = node_loc.size(0)
        edge_index = pyg.nn.knn_graph(x=node_loc, k=self.K_Main, loop=True)
        
        # edge feature
        edge_displacement = (torch.gather(node_loc, dim=0, index=edge_index[0].unsqueeze(-1).expand(-1,3)) - 
                             torch.gather(node_loc, dim=0, index=edge_index[1].unsqueeze(-1).expand(-1,3)))
        
        #Let's normalize edge_displacement
        eds = edge_displacement.shape #(n_node*self.K_Main, 3)
        assert eds == (n_node*self.K_Main, 3)
        
        edge_distance = torch.linalg.norm(edge_displacement, dim=1)
        assert edge_distance.shape == (eds[0],) #(n_node*self.K_Main,)
        
        edge_distance = torch.reshape(edge_distance, 
                                      shape=(n_node, self.K_Main))
        
        max_dist_pen = torch.max(edge_distance, 
                                 dim=1)[0] #max_distance_per_each_node
        
        max_dist_pen = expand(max_dist_pen, self.K_Main)
        edge_displacement_norm = torch.divide(edge_displacement, 
                                         max_dist_pen.unsqueeze(dim=1))
        
        if self.Is_Normalize:
            edge_displacement = F.normalize(edge_displacement)
        
        # return the graph with features
        graph = pyg.data.Data(
            edge_index=edge_index,
            edge_attr=edge_displacement_norm, #
            pos=node_loc
        )
        if self.Y is not None:
            graph.y = self.Y[idx]
        return graph

#Define graph dataset for train
class TorusDataset(pyg.data.Dataset):
    def __init__(self, K_Main, Is_Normalize):   
        super().__init__()
        self.X = []
        self.Y = []
        self.K_Main = K_Main
        self.Is_Normalize = Is_Normalize
        
        assert type(Is_Normalize) == bool
        for k in range(1,200+1):
            kth_X_path = '../../dataset/Section_31_TorusDataset/Position_Vectors/torus_{}_Position.npy'.format(k)
            kth_X = torch.from_numpy(np.load(kth_X_path)).float()
            self.X.append(kth_X)
            
            kth_Y_path = '../../dataset/Section_31_TorusDataset/Normal_Vectors/torus_{}_Normal.npy'.format(k)
            kth_Y = torch.from_numpy(np.load(kth_Y_path)).float()
            self.Y.append(kth_Y)
            
        for k in range(1,200+1):
            kth_X_path = '../../dataset/Section_31_TorusDataset/Position_Vectors/torus_cut_{}_Position.npy'.format(k)
            kth_X = torch.from_numpy(np.load(kth_X_path)).float()
            self.X.append(kth_X)
            
            kth_Y_path = '../../dataset/Section_31_TorusDataset/Normal_Vectors/torus_cut_{}_Normal.npy'.format(k)
            kth_Y = torch.from_numpy(np.load(kth_Y_path)).float()
            self.Y.append(kth_Y)
            
    def len(self):
        return len(self.X)
    
    def get(self, idx):
        node_loc = self.X[idx]
        #torch.manual_seed(self.seed_number)
        rand_ratio = (torch.rand(1)*4+1)[0]
        rand_idx = torch.randperm(len(node_loc))[:int(len(node_loc)/rand_ratio)]
        node_loc = node_loc[rand_idx]
                
#         noise_level = abs(node_loc).max() * .01
#         noise = torch.rand(node_loc.size())*2*noise_level - noise_level
#         node_loc += noise
        
        # construct the graph using the KNN
        n_node = node_loc.size(0)
        edge_index = pyg.nn.knn_graph(x=node_loc, k=self.K_Main, loop=True)
        
        # edge feature
        edge_displacement = (torch.gather(node_loc, dim=0, index=edge_index[0].unsqueeze(-1).expand(-1,3)) - 
                             torch.gather(node_loc, dim=0, index=edge_index[1].unsqueeze(-1).expand(-1,3))) 
        
        #Let's normalize edge_displacement
        eds = edge_displacement.shape #(n_node*self.K_Main, 3)
        assert eds == (n_node*self.K_Main, 3)
        
        edge_distance = torch.linalg.norm(edge_displacement, dim=1)
        assert edge_distance.shape == (eds[0],) #(n_node*self.K_Main,)
        
        edge_distance = torch.reshape(edge_distance, 
                                      shape=(n_node, self.K_Main))
        
        max_dist_pen = torch.max(edge_distance, 
                                 dim=1)[0] #max_distance_per_each_node
        
        max_dist_pen = expand(max_dist_pen, self.K_Main)
        edge_displacement_norm = torch.divide(edge_displacement, 
                                         max_dist_pen.unsqueeze(dim=1))
        
        if self.Is_Normalize:
            edge_displacement = F.normalize(edge_displacement)
        
        # return the graph with features
        graph = pyg.data.Data(
            edge_index=edge_index,
            edge_attr=edge_displacement_norm, #
            pos=node_loc
        )
        
        if self.Y is not None:
            graph.y = self.Y[idx][rand_idx]
        return graph