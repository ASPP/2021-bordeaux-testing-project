import logistic_func
import pytest
import numpy as np
@pytest.mark.parametrize('x,r, expected',[(0.1,2.2,0.198),(0.2,3.4,0.544),(0.75,1.7,0.31875)])
def test_logistic(x,r, expected):
    np.testing.assert_almost_equal(logistic_func.logistic(x,r) ,expected)


#x=0.1, r=2.2, it=1 => iterate_f(it, x, r)=[0.198]
#x=0.2, r=3.4, it=4 => f(x, r)=[0.544, 0.843418, 0.449019, 0.841163]
#x=0.75, r=1.7, it=2 => f(x, r)=[0.31875, 0.369152]]


@pytest.mark.parametrize('it,x,r,expected',[(4,0.2,3.4,[0.544, 0.843418, 0.449019, 0.841163])])
def test_iterate(it,x,r,expected):
    np.testing.assert_almost_equal(logistic_func.iterate_f(it,x,r) ,expected)