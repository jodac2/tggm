

def to_list(x):
    """
    To list
    """
    
    if x is None:
        return []
    elif type(x) is list:
        return x
    else:
        return [x]
