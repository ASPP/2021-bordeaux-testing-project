def logistic_map(x,r):
    return r*x*(1-x)

def iterate_f(x0, r, it):
    x = x0
    output = []
    for i in range(it):
        x = logistic_map(x, r)
        output.append(x)
    return output
