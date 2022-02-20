import numpy as np
import tensorly as tl
from tggm.utils.utils import to_list


def random_tensor(size=1, mu=None, sigma=None, shape=None, seed=None, stack=False):
    """
    Random tensor-normal
    ------
        Generated random tensor-normal(mu, sigma)
    
    Params
    ------
    size: size of sample
    sigma: list with covariance matrices of shape di x di, i=1, ..., n
    mu: tensor mean of shape d1 x ... x dp 
    shape: list with shape of random tensor, when mu and signa are None
    seed: seed to simulation
    stack: True is sample is stacked
    
    Return
    ------
    rtensor: list with tensors d1 x ... x dp if stack is False 
        else tensor size x d1 x ... x dp
    """
    
    if sigma is None and mu is None and shape is None:
        raise Exception("Any of sigma, mu or shape should be not None")
    
    if size < 1:
        raise Exception("Sample size should be > 0")
    
    if sigma is not None:
        
        sigma = to_list(sigma)

        shape = [si.shape[0] for si in sigma]
        if mu is not None and shape != list(mu.shape):
            raise Exception("Shape of sigma and mu should be equal")
        
    elif sigma is None and mu is not None:
        
        shape = list(mu.shape)
        
    else:
        
        if type(shape) not in [list, tuple]:
            raise Exception("Type of shape should be list")
    
    np.random.seed(seed)
    
    # Get A such S=AA'
    if sigma is not None:
        A = [np.linalg.cholesky(si) for si in sigma]
    
    # v ~ N(0, 1)
    rvector = np.random\
        .normal(size=np.prod(shape) * size)\
        .reshape((size, *shape))
    
    # T ~ TN(0, 1)
    rtensor = tl.tensor(rvector)
    rtensor = [rtensor[i] for i in range(size)]
    
    # T ~ TN(0, sigma)
    if sigma is not None:
        rtensor = [tl.tenalg.multi_mode_dot(ti, A) for ti in rtensor]
    
    # T ~ TN(mu, sigma)
    if mu is not None:
        rtensor = [ti + mu for ti in rtensor]
    
    # T ~ TN(mu, I + sigma)
    if stack:
        rtensor = tl.stack(rtensor, axis=0)
    
    np.random.seed(None)
    
    return rtensor


def random_matrix(size=1, mu=None, sigma=None, phi=None, shape=None, seed=None, stack=False):
    """
    Random matrix-normal
    ------
        Generated random matrix-normal(mu,  K(sigma), K(phi))
        
    Params
    ------
    size: size of sample
    sigma: list with covariance matrices of shape ci x ci, i=1, ..., m
        covariance cols = K(sigma)
    phi: list with covariance matrices of shape ri x ri, i=1, ..., n
        covariance rows = K(phi)
    mu: matrix mean of shape (r1 ... rn) x (c1 ... cm) 
    shape: list with shape of random matrix, when mu and (signa, phi) are None
    seed: seed to simulation
    stack: True is sample is stacked
    
    Return
    ------
    rmatrix: list with matrix (r1 ... rn) x (c1 ... cm) if stack is False 
        else tensor size x (r1 ... rn) x (c1 ... cm)
    """
    
    if size < 1:
        raise Exception("Sample size should be > 0")
    
    if sigma is None and phi is None and mu is None and shape is None:
        raise Exception("Any of (sigma, phi), mu or shape should be not None")
    
    if (sigma is None and phi is not None) or (sigma is None and phi is not None):
        raise Exception("sigma is None if only if phi is None")
        
    if sigma is not None:
        
        tsigma = to_list(sigma) + to_list(phi)
        
        nrow = np.prod([si.shape[0] for si in sigma])
        ncol = np.prod([si.shape[0] for si in phi])
        shape = [nrow, ncol]
        
        if mu is not None and shape != list(mu.shape):
            raise Exception("Shape of (sigma, phi) and mu should be equal")
        
    else:
        tsigma = None
    
    if sigma is None and mu is not None:
        shape = list(mu.shape)
    
    # T ~ TN(0, sigma)
    rtensor = random_tensor(
        size=size, 
        sigma=tsigma, 
        shape=shape,
        seed=seed
    )
    
    # M ~ MN(0, sigma, phi)
    rmatrix = [tl.unfold(Ti, 0).reshape(shape) for Ti in rtensor]
    
    # M ~ MN(mu, sigma, phi)
    if mu is not None:
        rmatrix = [Mi + mu for Mi in rmatrix]
    
    # T ~ TN(mu, I + sigma)
    if stack:
        rmatrix = tl.stack(rmatrix, axis=0)
    
    return rmatrix


def random_vector(size=1, mu=None, sigma=None, shape=None, seed=None, stack=False):
    """
    Random vector-normal
    ------
        Generated random vector-normal(mu,  K(sigma))
        
    Params
    ------
    size: size of sample
    sigma: list with covariance matrices of shape di x di, i=1, ..., n
        covariance = K(sigma)
    mu: vector mean of shape d1 ... dn 
    shape: list with shape of random vector, when mu and signa is None
    seed: seed to simulation
    stack: True is sample is stacked
    
    Return
    ------
    rvector: list with vectors d1 ... dn if stack is False 
        else matrix size x d1 ... dn
    """
    
    if size < 1:
        raise Exception("Sample size should be > 0")
    
    if sigma is None and mu is None and shape is None:
        raise Exception("Any of sigma, mu or shape should be not None")
        
    if sigma is not None:
        
        sigma = to_list(sigma)
        
        nrow = np.prod([si.shape[0] for si in sigma])
        shape = [nrow, ]
        
        if mu is not None and shape != list(mu.shape):
            raise Exception("Shape of (sigma, phi) and mu should be equal")
        
    if sigma is None and mu is not None:
        shape = list(mu.shape)
    
    # T ~ TN(0, sigma)
    rtensor = random_tensor(
        size=size, 
        sigma=sigma, 
        shape=shape,
        seed=seed
    )
    
    # V ~ N(0, sigma)
    rvector = [tl.unfold(Ti, 0).reshape(shape) for Ti in rtensor]
    
    # V ~ N(mu, sigma)
    if mu is not None:
        rvector = [Vi + mu for Vi in rvector]
    
    # M ~ MN(mu, I, sigma)
    if stack:
        rvector = tl.stack(rvector, axis=0)
    
    return rvector
