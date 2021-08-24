import numpy as np
import math
import pytest

import logistic_func as lf

@pytest.mark.parametrize("x, r, expected", [(0.1, 2.2, 0.198), (0.2, 3.4, 0.544), (0.75, 1.7, 0.31875)])
def test_logistic_map(x, r, expected):
    result = lf.logistic_map(x, r)
    assert np.isclose(result, expected)


@pytest.mark.parametrize("it, x, r, expected", 
                         [(1, 0.1, 2.2, [0.198]), 
                          (4, 0.2, 3.4, [0.544, 0.843418, 0.449019, 0.841163]), 
                          (2, 0.75, 1.7, [0.31875, 0.369152])])
def test_iterate_f(it, x, r, expected):
    result = lf.iterate_f(it, x, r)
    assert np.allclose(result, expected)





