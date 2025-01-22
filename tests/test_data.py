from src.python_uv.data import dict_operations, list_operations, tuple_operations


def test_list_operations() -> None:
    result = list_operations()
    assert len(result) == 8
    assert result[-1] == 8


def test_dict_operations() -> None:
    result = dict_operations()
    assert len(result) == 4
    assert result["grape"] == 4


def test_tuple_operations() -> None:
    result = tuple_operations()
    assert len(result) == 3
    assert isinstance(result[1], str)
