"""
An example test case with pytest.
See: https://docs.pytest.org/en/6.2.x/index.html
"""
# content of test_sample.py
def inc(x):
    return x 


def test_answer():
    assert inc(3) == 3
