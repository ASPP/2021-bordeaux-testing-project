import pytest
from math import isclose

from logistic import logistic_model


@pytest.mark.parametrize("x, r, expected",[(0.1,2.2,0.198),
                            (0.2,3.4,0.544),(0.75,1.7,0.31875)])
def test_logistic_model(x,r,expected):
    assert isclose(logistic_model(x,r), expected)
