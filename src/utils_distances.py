import torch

class NotImplementedError(Exception):
    pass 

def gromov_cost(xs,xt,T,device):
    C1 = _cost_matrix(xs,xs)
    C2 = _cost_matrix(xt,xt)
    p=torch.ones(xs.shape[0])/xs.shape[0]
    q=torch.ones(xt.shape[0])/xt.shape[0]
    p=p.to(device)
    q=q.to(device)  
    constC, hC1, hC2 = init_matrix(C1, C2, p, q,device, 'square_loss')
    
    tens=tensor_product(constC, hC1, hC2, T).to(device)
    
    return torch.sum(tens*T)

def _cost_matrix(x, y, p=2):
    "Returns the matrix of $|x_i-y_j|^p$."
    x_col = x.unsqueeze(-2)
    y_lin = y.unsqueeze(-3)
    C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
    return C
    
def entropic_gw(xs,xt,device,eps=1e-3,max_iter=100,verbose=False,log=True):

    C1 = _cost_matrix(xs,xs)
    C2 = _cost_matrix(xt,xt)
    p=torch.ones(xs.shape[0])/xs.shape[0]
    q=torch.ones(xt.shape[0])/xt.shape[0]
    p=p.to(device)
    q=q.to(device)
    if log:
        T,log=entropic_gromov_wasserstein(C1,C2,p,q,epsilon=eps,max_iter=max_iter,loss_fun='square_loss',verbose=verbose,device=device,log=True)
        return T,log
    else:
        T=entropic_gromov_wasserstein(C1,C2,p,q,epsilon=eps,max_iter=max_iter,loss_fun='square_loss',verbose=verbose,device=device,log=True)
        return T
    
    
def gwloss(constC, hC1, hC2, T):
    """ Return the Loss for Gromov-Wasserstein
    The loss is computed as described in Proposition 1 Eq. (6) in [12].
    Parameters
    ----------
    constC : ndarray, shape (ns, nt)
           Constant C matrix in Eq. (6)
    hC1 : ndarray, shape (ns, ns)
           h1(C1) matrix in Eq. (6)
    hC2 : ndarray, shape (nt, nt)
           h2(C) matrix in Eq. (6)
    T : ndarray, shape (ns, nt)
           Current value of transport matrix T
    Returns
    -------
    loss : float
           Gromov Wasserstein loss
    References
    ----------
    .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
    "Gromov-Wasserstein averaging of kernel and distance matrices."
    International Conference on Machine Learning (ICML). 2016.
    """

    tens = tensor_product(constC, hC1, hC2, T)

    return torch.sum(tens * T)

def init_matrix(C1, C2, p, q, device,loss_fun='square_loss'):
    """ Return loss matrices and tensors for Gromov-Wasserstein fast computation
    Returns the value of \mathcal{L}(C1,C2) \otimes T with the selected loss
    function as the loss function of Gromow-Wasserstein discrepancy.
    The matrices are computed as described in Proposition 1 in [12]
    Where :
        * C1 : Metric cost matrix in the source space
        * C2 : Metric cost matrix in the target space
        * T : A coupling between those two spaces
    The square-loss function L(a,b)=(1/2)*|a-b|^2 is read as :
        L(a,b) = f1(a)+f2(b)-h1(a)*h2(b) with :
            * f1(a)=(a^2)/2
            * f2(b)=(b^2)/2
            * h1(a)=a
            * h2(b)=b
    Parameters
    ----------
    C1 : ndarray, shape (ns, ns)
         Metric cost matrix in the source space
    C2 : ndarray, shape (nt, nt)
         Metric costfr matrix in the target space
    T :  ndarray, shape (ns, nt)
         Coupling between source and target spaces
    p : ndarray, shape (ns,)
    Returns
    -------
    constC : ndarray, shape (ns, nt)
           Constant C matrix in Eq. (6)
    hC1 : ndarray, shape (ns, ns)
           h1(C1) matrix in Eq. (6)
    hC2 : ndarray, shape (nt, nt)
           h2(C) matrix in Eq. (6)
    References
    ----------
    .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
    "Gromov-Wasserstein averaging of kernel and distance matrices."
    International Conference on Machine Learning (ICML). 2016.
    """

    if loss_fun == 'square_loss':
        def f1(a):
            return (a**2) / 2

        def f2(b):
            return (b**2) / 2

        def h1(a):
            return a

        def h2(b):
            return b
    elif loss_fun == 'kl_loss':
        raise NotImplementedError('Wait for it')

    constC1 = torch.matmul(torch.matmul(f1(C1), p.reshape(-1, 1)),
                     torch.ones(len(q)).to(device).reshape(1, -1))
    constC2 = torch.matmul(torch.ones(len(p)).to(device).reshape(-1, 1),
                     torch.matmul(q.reshape(1, -1), torch.transpose(f2(C2),1,0)))
    constC = constC1 + constC2
    hC1 = h1(C1)
    hC2 = h2(C2)

    return constC, hC1, hC2
    
    
def tensor_product(constC, hC1, hC2, T):
    """ Return the tensor for Gromov-Wasserstein fast computation
    The tensor is computed as described in Proposition 1 Eq. (6) in [12].
    Parameters
    ----------
    constC : ndarray, shape (ns, nt)
           Constant C matrix in Eq. (6)
    hC1 : ndarray, shape (ns, ns)
           h1(C1) matrix in Eq. (6)
    hC2 : ndarray, shape (nt, nt)
           h2(C) matrix in Eq. (6)
    Returns
    -------
    tens : ndarray, shape (ns, nt)
           \mathcal{L}(C1,C2) \otimes T tensor-matrix multiplication result
    References
    ----------
    .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
    "Gromov-Wasserstein averaging of kernel and distance matrices."
    International Conference on Machine Learning (ICML). 2016.
    """
    A = -torch.matmul(hC1, T).matmul(torch.transpose(hC2,1,0))
    tens = constC + A
    return tens
    
def gwggrad(constC, hC1, hC2, T):
    """ Return the gradient for Gromov-Wasserstein
    The gradient is computed as described in Proposition 2 in [12].
    Parameters
    ----------
    constC : ndarray, shape (ns, nt)
           Constant C matrix in Eq. (6)
    hC1 : ndarray, shape (ns, ns)
           h1(C1) matrix in Eq. (6)
    hC2 : ndarray, shape (nt, nt)
           h2(C) matrix in Eq. (6)
    T : ndarray, shape (ns, nt)
           Current value of transport matrix T
    Returns
    -------
    grad : ndarray, shape (ns, nt)
           Gromov Wasserstein gradient
    References
    ----------
    .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
    "Gromov-Wasserstein averaging of kernel and distance matrices."
    International Conference on Machine Learning (ICML). 2016.
    """
    return 2 * tensor_product(constC, hC1, hC2,
                              T)  # [12] Prop. 2 misses a 2 factor
    
    
def entropic_gromov_wasserstein(C1, C2, p, q, loss_fun, epsilon,device,
                            max_iter=100, tol=1e-6, verbose=False, log=False):
    """
    Returns the gromov-wasserstein transport between (C1,p) and (C2,q)
    (C1,p) and (C2,q)
    The function solves the following optimization problem:
    .. math::
        \GW = arg\min_T \sum_{i,j,k,l} L(C1_{i,k},C2_{j,l})*T_{i,j}*T_{k,l}-\epsilon(H(T))
        s.t. \GW 1 = p
             \GW^T 1= q
             \GW\geq 0
    Where :
        C1 : Metric cost matrix in the source space
        C2 : Metric cost matrix in the target space
        p  : distribution in the source space
        q  : distribution in the target space
        L  : loss function to account for the misfit between the similarity matrices
        H  : entropy
    Parameters
    ----------
    C1 : ndarray, shape (ns, ns)
         Metric cost matrix in the source space
    C2 : ndarray, shape (nt, nt)
         Metric costfr matrix in the target space
    p :  ndarray, shape (ns,)
         distribution in the source space
    q :  ndarray, shape (nt,)
         distribution in the target space
    loss_fun :  string
        loss function used for the solver either 'square_loss' or 'kl_loss'
    epsilon : float
        Regularization term >0
    max_iter : int, optional
       Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    Returns
    -------
    T : ndarray, shape (ns, nt)
        coupling between the two spaces that minimizes :
            \sum_{i,j,k,l} L(C1_{i,k},C2_{j,l})*T_{i,j}*T_{k,l}-\epsilon(H(T))
    References
    ----------
    .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
    "Gromov-Wasserstein averaging of kernel and distance matrices."
    International Conference on Machine Learning (ICML). 2016.
    """

    T = p[:,None]*q[None,:]
    
    constC, hC1, hC2 = init_matrix(C1, C2, p, q,device, loss_fun)
    
    cpt = 0
    err = 1
    
    if log:
        log = {'err': []}
    
    while (err > tol and cpt < max_iter):
    
        Tprev = T
    
        # compute the gradient
        tens = gwggrad(constC, hC1, hC2, T)
    
        T = sinkhorn(p, q, tens,device, epsilon)
    
        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            
            err = torch.norm(T - Tprev).item()
    
            if log:
                log['err'].append(err)
    
            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format(
                        'It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))
    
        cpt += 1
        
    
    if log:
        log['loss'] = gwloss(constC, hC1, hC2, T)
        return T, log
    else:
        return T
        
def sinkhorn(p,q,C, device,epsilon=1e-3,threshold = 1e-1,numItermax=100):
            
    # Initialise approximation vectors in log domain
    u = torch.zeros_like(p).to(device)
    v = torch.zeros_like(q).to(device)

    # Stopping criterion
   
    # Sinkhorn iterations
    for i in range(numItermax): 
        u0, v0 = u, v
                    
        # u^{l+1} = a / (K v^l)
        K = _log_boltzmann_kernel(u, v, C,epsilon)
        u_ = torch.log(p + 1e-8) - torch.logsumexp(K, dim=1)
        u = epsilon * u_ + u
                    
        # v^{l+1} = b / (K^T u^(l+1))
        K_t = _log_boltzmann_kernel(u, v, C,epsilon).transpose(-2, -1)
        v_ = torch.log(q + 1e-8) - torch.logsumexp(K_t, dim=1)
        v = epsilon * v_ + v
        
        # Size of the change we have performed on u
        diff = torch.sum(torch.abs(u - u0), dim=-1) + torch.sum(torch.abs(v - v0), dim=-1)
        mean_diff = torch.mean(diff)
                    
        if mean_diff.item() < threshold:
            break
   
    # Transport plan pi = diag(a)*K*diag(b)
    K = _log_boltzmann_kernel(u, v, C,epsilon)
    pi = torch.exp(K)
    
    # Sinkhorn distance
    #cost = torch.sum(pi * C, dim=(-2, -1))

    return pi
    
def _log_boltzmann_kernel(u, v,C,epsilon):
    kernel = -C + u.unsqueeze(-1) + v.unsqueeze(-2)
    kernel /= epsilon
    return kernel

 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 17:39:12 2019

@author: vayer
"""

import torch
import time

class BadShapeError(Exception):
    pass 

def sgw_gpu(xs,xt,device,nproj=200,tolog=False,P=None):
    """ Returns SGW between xs and xt eq (4) in [1]. Only implemented with the 0 padding operator Delta
    Parameters
    ----------
    xs : tensor, shape (n, p)
         Source samples
    xt : tensor, shape (n, q)
         Target samples
    device :  torch device
    nproj : integer
            Number of projections. Ignore if P is not None
    P : tensor, shape (max(p,q),n_proj)
        Projection matrix. If None creates a new projection matrix
    tolog : bool
            Wether to return timings or not
    Returns
    -------
    C : tensor, shape (n_proj,1)
           Cost for each projection
    References
    ----------
    .. [1] Vayer Titouan, Chapel Laetitia, Flamary R{\'e}mi, Tavenard Romain
          and Courty Nicolas
          "Sliced Gromov-Wasserstein"
    Example
    ----------
    import numpy as np
    import torch
    from sgw_pytorch import sgw
    
    n_samples=300
    Xs=np.random.rand(n_samples,2)
    Xt=np.random.rand(n_samples,1)
    xs=torch.from_numpy(Xs).to(torch.float32)
    xt=torch.from_numpy(Xt).to(torch.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    P=np.random.randn(2,500)
    sgw_gpu(xs,xt,device,P=torch.from_numpy(P).to(torch.float32))
    """    
    if tolog:
        log={}

    if tolog: 
        st=time.time()
        xsp,xtp=sink_(xs,xt,device,nproj,P)
        ed=time.time()   
        log['time_sink_']=ed-st
    else:
        xsp,xtp=sink_(xs,xt,device,nproj,P)
    if tolog:    
        st=time.time()
        d,log_gw1d=gromov_1d(xsp,xtp,tolog=True)
        ed=time.time()   
        log['time_gw_1D']=ed-st
        log['gw_1d_details']=log_gw1d
    else:
        d=gromov_1d(xsp,xtp,tolog=False)
    
    if tolog:
        return d,log
    else:
        return d

        
        

def _cost(xsp,xtp,tolog=False):   
    """ Returns the GM cost eq (3) in [1]
    Parameters
    ----------
    xsp : tensor, shape (n, n_proj)
         1D sorted samples (after finding sigma opt) for each proj in the source
    xtp : tensor, shape (n, n_proj)
         1D sorted samples (after finding sigma opt) for each proj in the target
    tolog : bool
            Wether to return timings or not
    Returns
    -------
    C : tensor, shape (n_proj,1)
           Cost for each projection
    References
    ----------
    .. [1] Vayer Titouan, Chapel Laetitia, Flamary R{\'e}mi, Tavenard Romain
          and Courty Nicolas
          "Sliced Gromov-Wasserstein"
    """
    st=time.time()

    xs=xsp
    xt=xtp

    xs2=xs*xs
    xs3=xs2*xs
    xs4=xs2*xs2

    xt2=xt*xt
    xt3=xt2*xt
    xt4=xt2*xt2

    X=torch.sum(xs,0)
    X2=torch.sum(xs2,0)
    X3=torch.sum(xs3,0)
    X4=torch.sum(xs4,0)
    
    Y=torch.sum(xt,0)
    Y2=torch.sum(xt2,0)
    Y3=torch.sum(xt3,0)
    Y4=torch.sum(xt4,0)
    
    xxyy_=torch.sum((xs2)*(xt2),0)
    xxy_=torch.sum((xs2)*(xt),0)
    xyy_=torch.sum((xs)*(xt2),0)
    xy_=torch.sum((xs)*(xt),0)
    
            
    n=xs.shape[0]

    C2=2*X2*Y2+2*(n*xxyy_-2*Y*xxy_-2*X*xyy_+2*xy_*xy_)

    power4_x=2*n*X4-8*X3*X+6*X2*X2
    power4_y=2*n*Y4-8*Y3*Y+6*Y2*Y2

    C=(1/(n**2))*(power4_x+power4_y-2*C2)
        
        
    ed=time.time()
    
    if not tolog:
        return C 
    else:
        return C,ed-st


def gromov_1d(xs,xt,tolog=False): 
    """ Solves the Gromov in 1D (eq (2) in [1] for each proj
    Parameters
    ----------
    xsp : tensor, shape (n, n_proj)
         1D sorted samples for each proj in the source
    xtp : tensor, shape (n, n_proj)
         1D sorted samples for each proj in the target
    tolog : bool
            Wether to return timings or not
    fast: use the O(nlog(n)) cost or not
    Returns
    -------
    toreturn : tensor, shape (n_proj,1)
           The SGW cost for each proj
    References
    ----------
    .. [1] Vayer Titouan, Chapel Laetitia, Flamary R{\'e}mi, Tavenard Romain
          and Courty Nicolas
          "Sliced Gromov-Wasserstein"
    """
    
    if tolog:
        log={}
    
    st=time.time()
    xs2,i_s=torch.sort(xs,dim=0)
    
    if tolog:
        xt_asc,i_t=torch.sort(xt,dim=0) #sort increase
        xt_desc,i_t=torch.sort(xt,dim=0,descending=True) #sort deacrese
        l1,t1=_cost(xs2,xt_asc,tolog=tolog)
        l2,t2=_cost(xs2,xt_desc,tolog=tolog)
    else:
        xt_asc,i_t=torch.sort(xt,dim=0)
        xt_desc,i_t=torch.sort(xt,dim=0,descending=True)
        l1=_cost(xs2,xt_asc,tolog=tolog)
        l2=_cost(xs2,xt_desc,tolog=tolog)   
    toreturn=torch.mean(torch.min(l1,l2)) 
    ed=time.time()  
   
    if tolog:
        log['g1d']=ed-st
        log['t1']=t1
        log['t2']=t2
 
    if tolog:
        return toreturn,log
    else:
        return toreturn
            
def sink_(xs,xt,device,nproj=200,P=None): #Delta operator (here just padding)
    """ Sinks the points of the measure in the lowest dimension onto the highest dimension and applies the projections.
    Only implemented with the 0 padding Delta=Delta_pad operator (see [1])
    Parameters
    ----------
    xs : tensor, shape (n, p)
         Source samples
    xt : tensor, shape (n, q)
         Target samples
    device :  torch device
    nproj : integer
            Number of projections. Ignored if P is not None
    P : tensor, shape (max(p,q),n_proj)
        Projection matrix
    Returns
    -------
    xsp : tensor, shape (n,n_proj)
           Projected source samples 
    xtp : tensor, shape (n,n_proj)
           Projected target samples 
    References
    ----------
    .. [1] Vayer Titouan, Chapel Laetitia, Flamary R{\'e}mi, Tavenard Romain
          and Courty Nicolas
          "Sliced Gromov-Wasserstein"
    """  
    dim_d= xs.shape[1]
    dim_p= xt.shape[1]
    
    if dim_d<dim_p:
        random_projection_dim = dim_p
        xs2=torch.cat((xs,torch.zeros((xs.shape[0],dim_p-dim_d)).to(device)),dim=1)
        xt2=xt
    else:
        random_projection_dim = dim_d
        xt2=torch.cat((xt,torch.zeros((xt.shape[0],dim_d-dim_p)).to(device)),dim=1)
        xs2=xs
     
    if P is None:
        P=torch.randn(random_projection_dim,nproj)
    p=P/torch.sqrt(torch.sum(P**2,0,True))
    
    try:
    
        xsp=torch.matmul(xs2,p.to(device))
        xtp=torch.matmul(xt2,p.to(device))
    except RuntimeError as error:
        print('----------------------------------------')
        print('xs origi dim :', xs.shape)
        print('xt origi dim :', xt.shape)
        print('dim_p :', dim_p)
        print('dim_d :', dim_d)
        print('random_projection_dim : ',random_projection_dim)
        print('projector dimension : ',p.shape)
        print('xs2 dim :', xs2.shape)
        print('xt2 dim :', xt2.shape)
        print('xs_tmp dim :', xs2.shape)
        print('xt_tmp dim :', xt2.shape)
        print('----------------------------------------')
        print(error)
        raise BadShapeError
    
    return xsp,xtp