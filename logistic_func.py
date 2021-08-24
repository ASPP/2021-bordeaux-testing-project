import numpy as np
def logistic(x,r):
    #f(𝑥)=𝑟∗𝑥∗(1−𝑥)
    return r*x*(1-x)

def iterate_f(it,x,r):
    out=np.empty(it)
    out[0]=x
    for i in np.arange(it):
        out[i]=logistic(x,r)
        x=out[i]
    return out 