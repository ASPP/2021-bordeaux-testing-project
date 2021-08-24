
import numpy as np

def logistic_map(x, r):
    result = r * x * (1 - x)
    return result

def iterate_f(it, x_start, r):
    x = x_start
    results = []
    for i in range(it):
        x = logistic_map(x, r)
        results.append(x)
    return results

