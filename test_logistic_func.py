import numpy as np
import math
import pytest

import logistic_func as lf

@pytest.mark.parametrize("x, r, expected", [(0.1, 2.2, 0.198), (0.2, 3.4, 0.544), (0.75, 1.7, 0.31875)])
def test_logistic_map(x, r, expected):
    result = lf.logistic_map(x, r)
    assert np.isclose(result, expected)
    





