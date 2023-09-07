"""
An example test case with pytest.
See: https://docs.pytest.org/en/6.2.x/index.html
"""
# content of test_sample.py
def inc(x):
    return x 


def test_answer():
    assert inc(3) == 3


import sys
sys.path.append('../py_src')

from gravitational_waves import principal_axes



assert 1.0 == principal_axes(10.0,2.0,3.0)

