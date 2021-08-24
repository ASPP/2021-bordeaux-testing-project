import logistic_func
import pytest
import numpy as np
@pytest.mark.parametrize('x,r, expected',[(0.1,2.2,0.198),(0.2,3.4,0.544),(0.75,1.7,0.31875)])
def test_logistic(x,r, expected):
    np.testing.assert_almost_equal(logistic_func.logistic(x,r) ,expected)