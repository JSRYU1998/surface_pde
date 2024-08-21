import torch

def compute_grad(f, x):
    return_value =  torch.autograd.grad(f, x, create_graph=True, grad_outputs=torch.ones_like(f), allow_unused=True)[0]
    if return_value == None:
        return_value = torch.zeros_like(f)
    return return_value
        
# [a,b,c] -> [a,a,...,a,b,b,...,b,c,c,...,c]
def expand(ind, N): # ind : 1d torch array
    ind = ind.reshape(-1,1)
    ind = ind.repeat(1,N)
    ind = ind.reshape(-1)
    return ind

def rel_l2_error(u, u_pred):
    num = ((u - u_pred)**2).sum()**.5
    den = (u**2).sum()**.5
    return num / den
    
def rel_max_error(u, u_pred):
    num = max(abs(u - u_pred))
    den = max(abs(u))
    return num / den

def bmm3(a,b,c):
    return torch.bmm(a,torch.bmm(b,c))

#compute true value of Laplace_Beltrami Operator
def laplace_beltrami_torch(eval_u_torch, eval_phi_torch, x,y,z,t):
    u = eval_u_torch(x,y,z,t)
    u_x = compute_grad(u,x)
    u_y = compute_grad(u,y)
    u_z = compute_grad(u,z)
    
    grad_u = torch.stack([u_x,u_y,u_z],1)
    
    u_xx = compute_grad(u_x,x)
    u_xy = compute_grad(u_x,y)
    u_xz = compute_grad(u_x,z)
    
    u_yx = compute_grad(u_y,x)
    u_yy = compute_grad(u_y,y)
    u_yz = compute_grad(u_y,z)
    
    u_zx = compute_grad(u_z,x)
    u_zy = compute_grad(u_z,y)
    u_zz = compute_grad(u_z,z)
    
    lapla_u = (u_xx+u_yy+u_zz).reshape(-1,1)
    
    hess_u = torch.stack([u_xx,u_xy,u_xz,u_yx,u_yy,u_yz,u_zx,u_zy,u_zz],1).reshape(-1,3,3)
    
    phi = eval_phi_torch(x,y,z)
    
    phi_x = compute_grad(phi,x)
    phi_y = compute_grad(phi,y)
    phi_z = compute_grad(phi,z)
    
    phi_xx = compute_grad(phi_x,x)
    phi_xy = compute_grad(phi_x,y)
    phi_xz = compute_grad(phi_x,z)
    
    phi_yx = compute_grad(phi_y,x)
    phi_yy = compute_grad(phi_y,y)
    phi_yz = compute_grad(phi_y,z)
    
    phi_zx = compute_grad(phi_z,x)
    phi_zy = compute_grad(phi_z,y)
    phi_zz = compute_grad(phi_z,z)
    
    grad_phi = torch.stack([phi_x,phi_y,phi_z],1)
    norm_grad_phi = torch.sqrt(phi_x**2+phi_y**2+phi_z**2).reshape(-1,1)
    
    n = grad_phi / norm_grad_phi
    
    d_n_u = (grad_u*n).sum(1).reshape(-1,1)
    
    lapla_phi = (phi_xx+phi_yy+phi_zz).reshape(-1,1)
    
    hess_phi = torch.stack([
        phi_xx,phi_xy,phi_xz,
        phi_yx,phi_yy,phi_yz,
        phi_zx,phi_zy,phi_zz
    ],1).reshape(-1,3,3)
    
    nth_phin = bmm3(n.reshape(-1,1,3),hess_phi,n.reshape(-1,3,1)).reshape(-1,1)
    twoH = (lapla_phi - nth_phin)/norm_grad_phi
    nth_un = bmm3(n.reshape(-1,1,3),hess_u,n.reshape(-1,3,1)).reshape(-1,1)
    return lapla_u - twoH*d_n_u - nth_un

#get the values of f, where f = u_t - LaplaceBeltrami(u)
def eval_f_torch(eval_u_torch, eval_phi_torch, x,y,z,t):
    u = eval_u_torch(x,y,z,t)
    u_t = compute_grad(u,t).reshape(-1,1)
    return u_t - laplace_beltrami_torch(eval_u_torch,eval_phi_torch,x,y,z,t)

#Redefine above functions to get 'points' as inputs directly  
def laplace_beltrami_point(eval_u, eval_phi, points, t):
    x = points[:,0].clone()
    x.requires_grad = True
    y = points[:,1].clone()
    y.requires_grad = True
    z = points[:,2].clone()
    z.requires_grad = True
    return laplace_beltrami_torch(eval_u, eval_phi, x,y,z, t)
    
def eval_f_point(eval_u, eval_phi, points, t):
    x = points[:,0].clone()
    x.requires_grad = True
    y = points[:,1].clone()
    y.requires_grad = True
    z = points[:,2].clone()
    z.requires_grad = True
    return eval_f_torch(eval_u, eval_phi, x,y,z,t)
