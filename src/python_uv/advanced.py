from collections.abc import Generator


def fibonacci(n: int) -> Generator[int, None, None]:
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b


class Animal:
    def __init__(self, name: str):
        self.name = name

    def speak(self) -> str:
        raise NotImplementedError("Subclass must implement abstract method")


class Dog(Animal):
    def speak(self) -> str:
        return f"{self.name} says Woof!"


def use_context_manager(filename: str) -> str:
    with open(filename) as file:
        return file.read()
