from python_uv.main import calculate_sum, greet


def test_greet() -> None:
    assert greet("Alice") == "Hello, Alice!"


def test_calculate_sum() -> None:
    assert calculate_sum(1, 2, 3) == 6
    assert calculate_sum() == 0
    assert calculate_sum(10) == 10
