def greet(name: str) -> str:
    return f"Hello, {name}!"


def calculate_sum(*args: int) -> int:
    return sum(args)


if __name__ == "__main__":
    print(greet("World"))
