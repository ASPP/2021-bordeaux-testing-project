from logistic_map import *
import pytest
import numpy as np


x=[0.1, 0.2, 0.75]
r=[2.2, 3.4, 1.7]
it=[1,4,2]


@pytest.mark.parametrize('x,r,expected', 
	[
	(0.1, 2.2, 0.198),
	(0.2, 3.4, 0.544),
	(0.75, 1.7, 0.31875),])
	
def test_f(x,r,expected):
	result = log_nap(r,x)
	assert np.isclose(result,expected)
	


@pytest.mark.parametrize('x,r,it,expected', 
	[
	(0.1, 2.2, 1, 0.198),
	(0.2, 3.4, 4, 0.544),
	(0.75, 1.7, 2, 0.31875),])
	
	
	
def test_log_nap(x,r,it,expected):
    #when
    result = iterate_f(it=it, fn1=log_nap, r=r, x=x)
    #then
    if len(result)==1: 
        expected = [0.198]
    elif len(result)==4:
        expected = [0.544, 0.843418, 0.449019, 0.841163]
    elif len(result)==2:
        expected = [0.31875, 0.369152]
    assert np.all(np.isclose(expected,result))
    
    
    
    
    
def test_fuzzy():
	result=[]
	expected=1/3
	r=1.5
	it=1000
	SEED=42
	x=np.random.rand(1000,1)
	#print(x)
	
	for i in x:
		result = iterate_f(it=it, fn1=log_nap, r=r, x=i)
		#print (result[:-100])
		
	
		assert np.isclose(round(np.mean(result[:-100]),2),round(expected,2))
			
