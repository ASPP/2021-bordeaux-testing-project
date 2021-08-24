
def logistic_model(x,r):
    return r*x*(1-x)

def logistic_iteration(it,x,r):

    logistic_out = [x]
    for i in range(it):
        logistic_out.append(logistic_model(logistic_out[i],r))

    return logistic_out[1:]
