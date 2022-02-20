import numpy as np


def random_phi(d, randA=None, randD=None, rand="normal", params=None, params2=None, inv=False, seed=None):
    """
    Random Phi = Sigma^1
    ------
        Random Phi = A'(D^-1)A
    
    Params
    ------
    d: dimension of matrix
    randA: Generator to upper diagonal elements of matrix A
    randB: Genetator to diagonal elements of matrid D^-1
    rand: distribution of elements of matrix A if randA is None
        uniform and normal are possible
    params: params of rand genetator to matrix A
        -loc and scale to nomrmal
        -low and high to uniform
    params2: params of rand generator to matriz D
        -scale and shape
    inv: True is return Phi^-1
    seed: seed
    
    Return
    ------
    Phi: random matrix d x d
    """
    
    if d < 2:
        raise Exception("Dimesion should be greater than 1")
    
    if randA is None:
        
        if rand == "uniform":
            
            if params is None:
                params = dict(low=-0.1, high=0.1)
            else:
                if "low" not in params.keys() or "high" not in params.keys():
                    raise Exception("Param low or high not in params")
                else:
                    if params["low"] >= params["high"]:
                        raise Exception("Param low should be smaller than high")
            
            def randA(size):
                return np.random.uniform(size=size, **params)
        
        else:
            
            if params is None:
                params = dict(loc=0, scale=0.1)
            else:
                if "loc" not in params.keys() or "scale" not in params.keys():
                    raise Exception("Param loc or scale not in params")
                else:
                    if params["scale"] <= 0:
                        raise Exception("Param loc should be greater than 0")
            
            def randA(size):
                return np.random.normal(size=size, **params)
    
    if randD is None:
        
        if params2 is None:
            params2 = dict(scale=1.0, shape=10)
        else:
            if "scale" not in params2.keys() or "shape" not in params2.keys():
                raise Exception("Param scale or shape not in params2")
            else:
                if params2["scale"] <= 0 or params2["shape"] <= 0:
                    raise Exception("Params scale and shape should be greater than 0")
        
        def randD(size):
            return 1/np.random.gamma(size=size, **params2)
    
    np.random.seed(seed)
    
    # Random matrix A
    A = np.identity(d)
    free_size = int(d * (d - 1)/2)
    free_indices = np.triu_indices(d, 1)
    A[free_indices] = randA(free_size)
    
    # Random matrix D^-1
    D = np.identity(d)
    np.fill_diagonal(D, randD(d))
    
    # Random matrix Phi
    Phi = A.T @ D @ A
    
    if inv:
        Phi = np.linalg.inv(Phi)
    
    np.random.seed(None)
    
    return Phi
