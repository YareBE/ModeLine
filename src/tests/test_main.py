import pytest
from src.main_test import sum ,is_greater_than

def test_sum():
    assert sum(2,5) == 7


def test_is_greater_than():
    assert is_greater_than(10,2)

@pytest.mark.parametrize(
        "input_x, input_y, expected",
        [
            (5,1,6), 
            (3,sum(1,0),4), 
            (8,9,17)
        ]
)
def test_sum_param(input_x, input_y, expected):
    assert sum(input_x, input_y) == expected