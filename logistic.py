
def logistic_model(x,r):
    return r*x*(1-x)

def logistic_iteration(it,x,r):

    logistic_out = [x]
    for i in range(it):
        logistic_out.append(logistic_model(logistic_out[i],r))

    return logistic_out[1:]

if __name__ == "__main__":
    x = [0.3,0.2,0.4]
    # x = 0.5
    it = 100
    r = 1.5
    log = logistic_iteration(it,x,r)
    print(log)
