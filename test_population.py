import pytest
import population
import math

@pytest.mark.parametrize('x,r,f', [(0.1, 2.2, 0.198),(0.2, 3.4, 0.544),(0.75, 1.7, 0.31875)])
def test_population(x,r,f):
    output = population.logistic_map(x,r)
    expected = f
    assert math.isclose(output, expected)
