def log_nap(r,x):
    y=r * x* (1-x)
    return y

def iterate_f(it,fn1,r,x):
    ylist=[fn1(r,x)]
    for i in range(1,it):
        ylist.append(fn1(r,x=ylist[i-1]))

    return ylist

if __name__=='__main__':
    temp_ylist = iterate_f(100, log_nap, 1, 0.5)
    print(temp_ylist)
