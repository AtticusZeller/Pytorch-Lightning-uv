def list_operations() -> list[int]:
    numbers = [1, 2, 3, 4, 5]
    numbers.append(6)
    numbers.extend([7, 8])
    return numbers


def dict_operations() -> dict[str, int]:
    fruits = {"apple": 1, "banana": 2, "orange": 3}
    fruits["grape"] = 4
    return fruits


def tuple_operations() -> tuple[int, str, bool]:
    return (1, "hello", True)
