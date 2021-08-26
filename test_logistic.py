from numpy.testing import assert_array_almost_equal, assert_allclose
import pytest
from math import isclose
import numpy as np
from logistic import logistic_model, logistic_iteration#


SEED = 42
random_state = np.random.RandomState(SEED)

@pytest.mark.parametrize("x, r, expected",[(0.1,2.2,0.198),
                            (0.2,3.4,0.544),(0.75,1.7,0.31875)])
def test_logistic_model(x,r,expected):
    assert isclose(logistic_model(x,r), expected)


@pytest.mark.parametrize("it, x, r, expected",[(1,0.1,2.2,[0.198]),
                            (4,0.2,3.4,[0.544, 0.843418, 0.449019, 0.841163]),
                            (2,0.75,1.7,[0.31875, 0.369152])])
def test_logistic_iteration(it,x,r,expected):
    result = logistic_iteration(it,x,r)
    assert_array_almost_equal(result, expected)

def test_fuzzing_converges():
    r = 1.5
    it = 100
    for _ in range(100):
        x=random_state.rand(1)
        result = logistic_iteration(it,x,r)[-1]
        assert_allclose(result, 1/3)
