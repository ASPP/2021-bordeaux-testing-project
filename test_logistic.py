from numpy.testing import assert_array_almost_equal
import pytest
from math import isclose

from logistic import logistic_model, logistic_iteration


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
