# -*- coding: utf-8 -*-

import torch
import torch_geometric as pyg
from torch import cos, sin, atan
from sklearn.neighbors import NearestNeighbors

####
def expand(ind, N): # ind : 1d torch array
    ind = ind.reshape(-1,1)
    ind = ind.repeat(1,N)
    ind = ind.reshape(-1)
    return ind
####

def align_normal(points, normals):
    nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(points)
    _, indices = nbrs.kneighbors(points)

#     print(indices.shape)

    n = len(points)

    visited = [False for _ in range(n)]

    stack = []
    stack.append(0)
    visited[0] = True

    while len(stack)>0:
        p = stack.pop()
#         print(p)

        for q in indices[p]:
            if not visited[q]:
                stack.append(q)
                visited[q] = True
                normals[q] *= torch.sign(torch.dot(normals[p],normals[q]))
    return normals

class PCALocalApproximation:
    def __init__(self, args, X):
        # X: point clouds of size (N, 3)
        self.args = args
        self.X = X
        
        self.find_local_coordinate()
        self.set_basis_function(extra=True)
        self.compute_surface_coefficient()
                        
    def find_local_coordinate(self):
        # find K-nearest neighbors for each point
        K = self.args.K
        nbrs = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(self.X)
        distances, self.indices = nbrs.kneighbors(self.X)
        distances = torch.FloatTensor(distances)

        self.X_knn = self.X[self.indices]

        # Wendland weight function
        D = 1.1 * distances.max()
        weight_eval = lambda d: (1 - d/D)**4 * (4*d/D + 1)
        self.weight = weight_eval(distances)

#         self.weight = torch.ones((len(self.X), K))
#         self.weight[:,1:] /= K

        # apply PCA to the covariance matrix associated with each neighborhood
        centers = self.X_knn.mean(1)
        diff = self.X_knn - centers.reshape(-1,1,3)
        covariances = (diff.reshape(-1,K,3,1) * diff.reshape(-1,K,1,3)).sum(1)
        eigval, eigvec = torch.linalg.eig(covariances)
        eigval = eigval.real
        eigvec = eigvec.real

        # the eigenvector corresponding to the smallest eigenvalue -> the normal vector
        # the others -> the basis for the tangent space 
        normal_indices = eigval.argmin(1)
        tmp = torch.gather(eigvec, dim=2, index=normal_indices.reshape(-1,1,1).expand(-1,3,1)).squeeze()
        self.normal_vectors = tmp / torch.linalg.norm(tmp, axis=-1).reshape(-1,1)
        
        tmp = torch.stack([torch.stack([vec[:,ind] for ind in range(3) if ind != normal_indices[i]]) for i, vec in enumerate(eigvec)])
        tmp[:,0] = tmp[:,0] / torch.linalg.norm(tmp[:,0], axis=-1).reshape(-1,1)
        tmp[:,1] = tmp[:,1] / torch.linalg.norm(tmp[:,1], axis=-1).reshape(-1,1)
        self.tangent_vectors = tmp.permute(0,2,1)
        

        
    def set_basis_function(self, extra=False):
        # local quadratic polynomial basis function [1, x, y, x^2, xy, y^2]
        coord = torch.einsum('bij,bki->bkj', 
                             torch.concat((self.tangent_vectors, self.normal_vectors.reshape(-1,3,1)), -1), 
                             self.X_knn - self.X.reshape(-1,1,3))
        
        self.basis = torch.stack((torch.ones_like(coord[...,0]), 
                                  coord[...,0],
                                  coord[...,1], 
                                  coord[...,0]**2, 
                                  coord[...,0]*coord[...,1], 
                                  coord[...,1]**2), -1)
        
        if extra:
#             coord_max = abs(coord[...,:-1]).max((1,2))
#             coord_max = coord_max.reshape(-1,1,1)
#             cmin = -coord_max/2
#             cmax = coord_max/2
            
            coord_max = abs(coord[...,:-1]).mean((1,2))
            coord_max = coord_max.reshape(-1,1,1)
            cmin = -coord_max
            cmax = coord_max
            
            self.extra_coord = torch.rand(len(self.X),20,2) * (cmax - cmin) + cmin
            self.extra_basis = torch.stack((torch.ones((len(self.X),20)),
                                            self.extra_coord[...,0], 
                                            self.extra_coord[...,1], 
                                            self.extra_coord[...,0]**2, 
                                            self.extra_coord[...,0]*self.extra_coord[...,1], 
                                            self.extra_coord[...,1]**2), -1)
        
        # the left hand side of the linear equation
        K = self.args.K
        self.A = (self.weight.reshape(-1,K,1,1) * self.basis.reshape(-1,K,6,1) * self.basis.reshape(-1,K,1,6)).sum(1)
        self.target_z = coord[...,2]
        
    def compute_surface_coefficient(self):
        K = self.args.K
        b = (self.weight.reshape(-1,K,1) * self.basis * self.target_z.reshape(-1,K,1)).sum(1)
        self.coef_a = torch.linalg.solve(self.A, b)
    
    def predict_surface(self):
        extra_z = (self.extra_basis * self.coef_a.reshape(-1,1,6)).sum(-1).reshape(-1,20,1)
        extra_coord = torch.concat((self.extra_coord, extra_z), -1)
        
        recons = torch.einsum('bji,bki->bkj', 
                              torch.concat((self.tangent_vectors, self.normal_vectors.reshape(-1,3,1)), -1), 
                              extra_coord)
        recons += self.X.reshape(-1,1,3)
        recons = recons.reshape(-1,3)
        return recons
    
    def compute_function_coefficient(self, target_u):
        # target_u: the evaluation on the point cloud
        K = self.args.K
        b = (self.weight.reshape(-1,K,1) * self.basis * target_u[self.indices]).sum(1)
        return torch.linalg.solve(self.A, b)
    
    def predict_function(self, coef_u):
        # coef_u: shape (len(X), 6)
        return (self.extra_basis * coef_u.reshape(-1,1,6)).sum(-1).reshape(-1)
    
    
    
class GNNLocalApproximation(PCALocalApproximation):
    def __init__(self, args, X, model):
        # X: point clouds of size (N, 3)
        self.args = args
        self.X = X
        self.model = model
        
        self.find_local_coordinate()
        self.set_basis_function(extra=True)
        self.compute_surface_coefficient()
        
    def find_local_coordinate(self):
        # find K-nearest neighbors for each point
        K = self.args.K
        nbrs = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(self.X)
        distances, self.indices = nbrs.kneighbors(self.X)
        distances = torch.FloatTensor(distances)

        self.X_knn = self.X[self.indices]
        
        # Wendland weight function
        D = 1.1 * distances.max()
        weight_eval = lambda d: (1 - d/D)**4 * (4*d/D + 1)
        self.weight = weight_eval(distances)
        
        with torch.no_grad():
            node_loc = self.X        
            n_node = node_loc.size(0)
            edge_index = pyg.nn.knn_graph(x=node_loc, k=K, loop=False)

            edge_displacement = (torch.gather(node_loc, dim=0, index=edge_index[0].unsqueeze(-1).expand(-1,3)) - 
                                 torch.gather(node_loc, dim=0, index=edge_index[1].unsqueeze(-1).expand(-1,3)))
            
            #Let's normalize edge_displacement
            eds = edge_displacement.shape #(n_node*self.K_Main, 3)
            assert eds == (n_node*K, 3)

            edge_distance = torch.linalg.norm(edge_displacement, dim=1)
            assert edge_distance.shape == (eds[0],) #(n_node*self.K_Main,)

            edge_distance = torch.reshape(edge_distance, 
                                          shape=(n_node, K))

            max_dist_pen = torch.max(edge_distance, 
                                     dim=1)[0] #max_distance_per_each_node

            max_dist_pen = expand(max_dist_pen, K)
            edge_displacement_norm = torch.divide(edge_displacement, 
                                             max_dist_pen.unsqueeze(dim=1))
            
            #edge_displacement = F.normalize(edge_displacement)

            data = pyg.data.Data(
                edge_index=edge_index,
                edge_attr=edge_displacement_norm, #
                pos=node_loc
            ).to(list(self.model.parameters())[0].data.device)

            normal_vectors = self.model(data).cpu()
            self.normal_vectors = align_normal(node_loc, normal_vectors)
            
            tmp = atan(-(self.normal_vectors[:,0] / self.normal_vectors[:,1]))
            tangent0 = torch.stack([cos(tmp), sin(tmp), torch.zeros_like(tmp)], 1)
            tangent1 = torch.cross(self.normal_vectors, tangent0)
            self.tangent_vectors = torch.stack((tangent0, tangent1), 2)
            
            
            
            
class DeepFitLocalApproximation(PCALocalApproximation):
    def __init__(self, args, X, normal_vectors):
        # X: point clouds of size (N, 3)
        # nv_pred : normal_vectors wrt GCS predicted by DeepFit (shape : (N,3))
        self.X = X
        self.normal_vectors = normal_vectors
        self.args = args
        
        self.find_local_coordinate()
        self.set_basis_function(extra=True)
        self.compute_surface_coefficient()
        
    def find_local_coordinate(self):
        # find K-nearest neighbors for each point
        K = self.args.K
        nbrs = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(self.X)
        distances, self.indices = nbrs.kneighbors(self.X)
        distances = torch.FloatTensor(distances)

        self.X_knn = self.X[self.indices]
        
        # Wendland weight function
        D = 1.1 * distances.max()
        weight_eval = lambda d: (1 - d/D)**4 * (4*d/D + 1)
        self.weight = weight_eval(distances)
        
        with torch.no_grad():            
            tmp = atan(-(self.normal_vectors[:,0] / self.normal_vectors[:,1]))
            tangent0 = torch.stack([cos(tmp), sin(tmp), torch.zeros_like(tmp)], 1)
            tangent1 = torch.cross(self.normal_vectors, tangent0)
            self.tangent_vectors = torch.stack((tangent0, tangent1), 2)
            
            
            
g11_eval = lambda coef_a: (1+coef_a[:,[2]]**2) / (1+coef_a[:,[1]]**2+coef_a[:,[2]]**2)
g12_eval = lambda coef_a: -coef_a[:,[1]]*coef_a[:,[2]] / (1+coef_a[:,[1]]**2+coef_a[:,[2]]**2)
g22_eval = lambda coef_a: (1+coef_a[:,[1]]**2) / (1+coef_a[:,[1]]**2+coef_a[:,[2]]**2)

surface_grad_eval = lambda g11, g12, g22, coef_a, coef_b, e1, e2, e3: \
    (g11*coef_b[:,[1]] + g12*coef_b[:,[2]])*(e1 + coef_a[:,[1]]*e3) + (g12*coef_b[:,[1]] + g22*coef_b[:,[2]])*(e2 + coef_a[:,[2]]*e3)

coef_A_eval = lambda g11, g12, g22, coef_a: \
    torch.concat([-2*coef_a[:,[1]]/(1+coef_a[:,[1]]**2+coef_a[:,[2]]**2) * (g11*coef_a[:,[3]] + g12*coef_a[:,[4]] + g22*coef_a[:,[5]]), 
                  -2*coef_a[:,[2]]/(1+coef_a[:,[1]]**2+coef_a[:,[2]]**2) * (g11*coef_a[:,[3]] + g12*coef_a[:,[4]] + g22*coef_a[:,[5]]), 
                  2*g11, 2*g12, 2*g22], 1)

class SurfaceDerivative:
    def __init__(self, coef_a, tangent_vectors, normal_vectors):
        self.coef_a = coef_a
        self.tangent_vectors = tangent_vectors
        self.normal_vectors = normal_vectors

        self.g11 = g11_eval(self.coef_a)
        self.g12 = g12_eval(self.coef_a)
        self.g22 = g22_eval(self.coef_a)
        
    def gradient(self, coef_u):
        return surface_grad_eval(self.g11, 
                                 self.g12, 
                                 self.g22, 
                                 self.coef_a, 
                                 coef_u, 
                                 self.tangent_vectors[...,0], 
                                 self.tangent_vectors[...,1], 
                                 self.normal_vectors)
        
    def laplacian(self, coef_u):
        coef_A = coef_A_eval(self.g11, 
                             self.g12, 
                             self.g22, 
                             self.coef_a)
        return (coef_A*coef_u[:,1:]).sum(1)
        
    def to(self, device):
        self.coef_a = self.coef_a.to(device)
        self.tangent_vectors = self.tangent_vectors.to(device)
        self.normal_vectors = self.normal_vectors.to(device)

        self.g11 = self.g11.to(device)
        self.g12 = self.g12.to(device)
        self.g22 = self.g22.to(device)
        return self
