from cmath import exp
from logging import raiseExceptions
from unittest import expectedFailure
import numpy as np
import tensorly as tl
from tggm.utils.utils import to_list


def random_sigma(d, s=1, seed=None):
    """
    Random positive definite matrix(d,d)
    ------
    
    Params
    ------
    d: int to dimension of matrix
    s: scale for variances
    seed: seed to random
    
    Return
    ------
    S: positive definite matrix(d,d)
    """
    
    if s <= 0:
        raise Exception("Value of s=%s shouldn't < 0" % s)
    
    if d < 1:
        raise Exception("Value of d=%s shouldn't < 1 " % d)
    
    np.random.seed(seed)
    
    # Simetric matrix
    A = np.random.uniform(size=d * d).reshape((d, d))
    A = 0.5 * (A + A.T)
    
    # Enssured positive definite
    A = A + d * np.identity(d)
    
    # Scale
    sm = A.diagonal().max()
    S = s * A/sm
    
    np.random.seed(None)
    
    return S


def random_tnorm(size=1, mu=None, sigma=None, shape=None, seed=None, stack=False):
    """
    Random tensor-normal
    ------
        Generated random tensor-normal(nu, sigma)
    
    Params
    ------
    size: size of sample
    sigma: list with covariance matriz of shape (di, di), i=1, ..., n
    mu: tensor mean of shape d1 x ... x dp 
    shape: list with shape of random tensor, when mu and signa is None
    seed: seed to simulation
    stack: True is sample is stacked
    
    Return
    ------
    tnorm: list with tensors d1 x ... x dp if stack is False 
        else tensor d1 x ... x dp x size
    """
    
    if sigma is None and mu is None and shape is None:
        raise Exception("Any of sigma, mu or shape should be not None")
    
    if size < 1:
        raise Exception("Sample size should be > 0")
    
    if sigma is not None and mu is not None:
        
        shape = [si.shape[0] for si in sigma]
        if shape != list(mu.shape):
            raise Exception("Shape of sigma and mu should be equal")
        
    elif sigma is not None and mu is None:
        
        shape = [si.shape[0] for si in sigma]

    elif sigma is None and mu is not None:
        
        shape = list(mu.shape)

    else:

        if type(shape) is not list:
            raise Exception("Type of shape should be list")
    
    np.random.seed(seed)
    
    # Get A such S=AA'
    if sigma is not None:
        A = [np.linalg.cholesky(si) for si in sigma]
    
    # x ~ N(0, 1)
    vnorm = np.random\
        .normal(size=np.prod(shape) * size)\
        .reshape((*shape, size))
    
    # t ~ TN(0, 1)
    tnorm = tl.tensor(vnorm)
    tnorm = [tnorm[..., i] for i in range(size)]
    
    # t ~ TN(0, sigma)
    if sigma is not None:
        tnorm = [tl.tenalg.multi_mode_dot(ti, A) for ti in tnorm]
    
    # t ~ TN(mu, sigma)
    if mu is not None:
        tnorm = [ti + mu for ti in tnorm]
    
    if stack:
        tnorm = tl.stack(tnorm, axis=len(shape))
    
    np.random.seed(None)
    
    return tnorm
